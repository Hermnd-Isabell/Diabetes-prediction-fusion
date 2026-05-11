import torch
from torch.utils.data import Dataset
import pandas as pd
import numpy as np
import re
from scipy import sparse
from scipy.sparse.linalg import spsolve
from scipy.signal import savgol_filter

# ----------------------------
# 默认向后兼容的预处理函数
# ----------------------------
def preprocess_spectrum(spectrum):
    """默认预处理：简单 min-max 归一化（向后兼容）"""
    spectrum = np.array(spectrum, dtype=np.float32)
    spectrum = (spectrum - np.min(spectrum)) / (np.max(spectrum) - np.min(spectrum) + 1e-8)
    return spectrum


# ----------------------------
# AsLS 基线校正 (Whittaker 平滑器)
# ----------------------------
def _baseline_als(y, lam=1e6, p=0.001, niter=10):
    """Asymmetric Least Squares (AsLS) 基线校正。
    参数:
        lam: 平滑度惩罚系数（越大越平滑）
        p: 不对称权重（0 < p < 1，越小对负偏差越敏感）
        niter: 迭代次数
    """
    L = len(y)
    # 预计算并转为 CSC 格式，消除 spsolve 的 SparseEfficiencyWarning
    D = sparse.diags([1.0, -2.0, 1.0], [0, -1, -2], shape=(L, L - 2))
    DDt = (D.dot(D.transpose())).tocsc()
    w = np.ones(L, dtype=np.float64)
    y64 = np.asarray(y, dtype=np.float64)
    for _ in range(niter):
        W = sparse.spdiags(w, 0, L, L).tocsc()
        Z = (W + lam * DDt).tocsc()
        z = spsolve(Z, w * y64)
        w = p * (y64 > z) + (1 - p) * (y64 < z)
    return z


# ----------------------------
# RamanDataset (兼容 gen_synthetic_data.py)
# ----------------------------
class RamanDataset(Dataset):
    def __init__(self, spectra_csv, clinical_csv, wave_cols,
                 label_col="Group", preprocess_fn=None,
                 min_scans=1, max_scans=180, augment=False,
                 aug_noise=0.01, aug_scale=0.1,
                 preprocess_cfg=None,
                 normalization_method="SNV",
                 scan_aggregation="sequence"):
        """
        spectra_csv: 光谱数据 (一行一个扫描)
        clinical_csv: 临床数据 (一行一个病人)
        wave_cols: 光谱波点列
        preprocess_fn: 向后兼容的旧式预处理函数（仅当 preprocess_cfg 为 None 时使用）
        preprocess_cfg: 配置驱动的预处理参数字典，格式:
            {
                "baseline_correction": {"enabled": true, "method": "AsLS", "lam": 1e6, "p": 0.001, "niter": 10},
                "smoothing": {"enabled": true, "window": 11, "polyorder": 2},
            }
        normalization_method: "SNV" (逐扫描归一化) | "patient_zscore" (患者级 Z-score)
        scan_aggregation: "sequence" (保留扫描序列，padding 到 max_scans) | "stats" (聚合为 mean/std/max)
        augment: 是否启用数据增强 (仅训练集)
        aug_noise: 高斯噪声标准差
        aug_scale: 强度缩放比例 (+/-)
        """
        self.spectra_df = pd.read_csv(spectra_csv, sep=None, engine="python")
        self.clin_df = pd.read_csv(clinical_csv, sep=None, engine="python")
        self.augment = augment
        self.aug_noise = aug_noise
        self.aug_scale = aug_scale
        self.max_scans = max_scans
        self.normalization_method = normalization_method
        self.scan_aggregation = scan_aggregation

        # 去除列名空格和BOM
        self.spectra_df.columns = self.spectra_df.columns.str.replace('\ufeff', '').str.strip()
        self.clin_df.columns = self.clin_df.columns.str.replace('\ufeff', '').str.strip()

        self.wave_cols = wave_cols
        self.label_col = label_col

        # 向后兼容：确定预处理策略的优先级
        if preprocess_cfg is not None:
            self._use_configurable = True
            self._preprocess_cfg = preprocess_cfg
            self._bc_cfg = self._preprocess_cfg.get("baseline_correction", {})
            self._sm_cfg = self._preprocess_cfg.get("smoothing", {})
            self._preprocess_fn = None
        elif preprocess_fn is not None:
            self._use_configurable = False
            self._preprocess_fn = preprocess_fn
            self._preprocess_cfg = None
        else:
            self._use_configurable = False
            self._preprocess_fn = None
            self._preprocess_cfg = None

        # 自动推断 label_col for Clinical Data
        self.clin_label_col = label_col
        if label_col not in self.clin_df.columns:
            if "Label" in self.clin_df.columns:
                self.clin_label_col = "Label"
            elif "Group" in self.clin_df.columns:
                self.clin_label_col = "Group"

        print(f"[DEBUG MSG] Config Label: {label_col}, Detected Clinical Label: {self.clin_label_col}")

        # reset 临床索引，方便 merge
        if "PatientID" in self.clin_df.columns:
            self.clin_df = self.clin_df.reset_index(drop=True)
        elif self.clin_df.index.name == "PatientID":
            self.clin_df = self.clin_df.reset_index()

        # 从光谱的 Sample 提取患者ID（如 "P1000-3" -> "P1000", "100-123" -> "100"）
        # copy() 避免宽 DataFrame 插入新列时的 PerformanceWarning
        self.spectra_df = self.spectra_df.copy()
        self.spectra_df["PatientID"] = self.spectra_df["Sample"].apply(
            lambda s: str(s).split('-')[0].replace('.txt', '')
        )

        # 确保 ID 类型一致
        self.spectra_df["PatientID"] = self.spectra_df["PatientID"].astype(str)
        self.clin_df["PatientID"] = self.clin_df["PatientID"].astype(str)

        # 匹配共同的 PatientID
        self.samples = sorted(list(
            set(self.spectra_df["PatientID"]).intersection(set(self.clin_df["PatientID"]))
        ))

        self.items = []
        for pid in self.samples:
            spec_rows = self.spectra_df[self.spectra_df["PatientID"] == pid]
            clin_row = self.clin_df[self.clin_df["PatientID"] == pid].iloc[0]

            # 原始光谱矩阵 [num_scans, num_wavelengths]
            raw_spectra = spec_rows[self.wave_cols].values.astype(np.float32)

            # === 预处理阶段（基线校正 + 平滑） ===
            processed = self._apply_preprocessing(raw_spectra)

            # === 归一化阶段 ===
            if self.normalization_method == "SNV":
                # 逐扫描 SNV
                normalized = np.stack([
                    (s - s.mean()) / (s.std() + 1e-8) for s in processed
                ])
            elif self.normalization_method == "patient_zscore":
                # 患者级 Z-score：利用该患者所有 scans 的分布抑制个体间基线漂移
                patient_mean = processed.mean(axis=0)          # [L]
                patient_std = processed.std(axis=0) + 1e-8      # [L]
                normalized = (processed - patient_mean) / patient_std
            else:
                normalized = processed

            # === 聚合 / 序列化 ===
            if self.scan_aggregation == "sequence":
                spectra_out, mask_out = self._to_sequence(normalized)
            elif self.scan_aggregation == "stats":
                spectra_out, mask_out = self._to_stats(normalized)
            else:
                raise ValueError(f"Unknown scan_aggregation: {self.scan_aggregation}")

            # 临床特征
            tabular = clin_row.drop(["PatientID", self.clin_label_col], errors="ignore").values.astype(np.float32)

            # 处理标签
            val = clin_row[self.clin_label_col]
            try:
                label = int(val)
            except:
                label = 1 if str(val).strip() == "DM" else 0

            # DEBUG: 打印前3个样本的临床数据
            if len(self.items) < 3:
                print(
                    f"[DEBUG MSG] Patient={pid}, Label={label}, Tabular Shape={tabular.shape}, "
                    f"Spectra Shape={spectra_out.shape}, Norm={self.normalization_method}, "
                    f"Aggregation={self.scan_aggregation}"
                )
                if self.normalization_method == "patient_zscore" and len(self.items) < 1:
                    pm = processed.mean(axis=0)
                    ps = processed.std(axis=0) + 1e-8
                    print(f"[DEBUG MSG] Patient {pid} Z-score stats -> mean[:5]={pm[:5]}, std[:5]={ps[:5]}")

            self.items.append({
                "spectra": spectra_out,
                "mask": mask_out,
                "tabular": tabular,
                "label": label
            })

    def _apply_preprocessing(self, raw_spectra):
        """应用预处理（基线校正 + 平滑），向后兼容旧式 preprocess_fn"""
        if not self._use_configurable:
            if self._preprocess_fn is not None:
                return np.stack([self._preprocess_fn(s) for s in raw_spectra])
            else:
                return raw_spectra

        processed = []
        for s in raw_spectra:
            s = s.copy()

            # 1. 基线校正
            if self._bc_cfg.get("enabled", False):
                if self._bc_cfg.get("method", "AsLS") == "AsLS":
                    baseline = _baseline_als(
                        s,
                        lam=self._bc_cfg.get("lam", 1e6),
                        p=self._bc_cfg.get("p", 0.001),
                        niter=self._bc_cfg.get("niter", 10)
                    )
                    s = s - baseline

            # 2. Savitzky-Golay 平滑
            if self._sm_cfg.get("enabled", False):
                window = self._sm_cfg.get("window", 11)
                polyorder = self._sm_cfg.get("polyorder", 2)
                if len(s) >= window:
                    s = savgol_filter(s, window_length=window, polyorder=polyorder)

            processed.append(s.astype(np.float32))
        return np.stack(processed)

    def _to_sequence(self, normalized):
        """序列模式：padding / truncate 到 max_scans，返回 [max_scans, L] + mask"""
        S, L = normalized.shape
        if S > self.max_scans:
            spectra = normalized[:self.max_scans]
        elif S < self.max_scans:
            pad_len = self.max_scans - S
            spectra = np.concatenate([
                normalized,
                np.zeros((pad_len, L), dtype=np.float32)
            ], axis=0)
        else:
            spectra = normalized

        mask = np.zeros(self.max_scans, dtype=bool)
        mask[:min(S, self.max_scans)] = True
        return spectra, mask

    def _to_stats(self, normalized):
        """统计聚合模式：mean / std / max -> [3 * L]，消除 padding 和可变长度"""
        mean_vec = normalized.mean(axis=0)       # [L]
        std_vec = normalized.std(axis=0) + 1e-8   # [L]
        max_vec = normalized.max(axis=0)         # [L]
        stats_vector = np.concatenate([mean_vec, std_vec, max_vec])  # [3*L]
        return stats_vector, None

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        item = self.items[idx]
        spectra = item["spectra"].copy()  # 复制以避免修改原始数据

        # 数据增强（对 sequence 模式的 [S, L] 或 stats 模式的 [3*L] 均有效）
        if self.augment:
            lower = 1.0 - self.aug_scale
            upper = 1.0 + self.aug_scale
            scale = np.random.uniform(lower, upper)
            spectra = spectra * scale

            noise = np.random.normal(0, self.aug_noise, size=spectra.shape).astype(np.float32)
            spectra = spectra + noise

        result = {
            "spectra": torch.tensor(spectra, dtype=torch.float32),
            "tabular": torch.tensor(item["tabular"], dtype=torch.float32),
            "label": torch.tensor(item["label"], dtype=torch.long)
        }
        # mask 仅在 sequence 模式下返回；stats 模式下不返回 mask 键
        if item["mask"] is not None:
            result["mask"] = torch.tensor(item["mask"], dtype=torch.bool)
        return result


# ----------------------------
# collate_fn（兼容 sequence 与 stats 两种模式）
# ----------------------------
def collate_fn(batch):
    spectra = torch.stack([b["spectra"] for b in batch])
    tabular = torch.stack([b["tabular"] for b in batch])
    labels = torch.stack([b["label"] for b in batch])

    result = {"spectra": spectra, "tabular": tabular, "label": labels}
    # 仅在存在 mask 时才 stack（stats 模式下无 mask）
    if "mask" in batch[0] and batch[0]["mask"] is not None:
        result["mask"] = torch.stack([b["mask"] for b in batch])
    return result
