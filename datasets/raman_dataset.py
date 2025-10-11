import torch
from torch.utils.data import Dataset
import pandas as pd
import numpy as np
import re

# ----------------------------
# 光谱预处理函数
# ----------------------------
def preprocess_spectrum(spectrum):
    spectrum = np.array(spectrum, dtype=np.float32)
    spectrum = (spectrum - np.min(spectrum)) / (np.max(spectrum) - np.min(spectrum) + 1e-8)
    return spectrum


# ----------------------------
# RamanDataset (兼容 gen_synthetic_data.py)
# ----------------------------
class RamanDataset(Dataset):
    def __init__(self, spectra_csv, clinical_csv, wave_cols,
                 label_col="Group", preprocess_fn=None,
                 min_scans=1, max_scans=180):
        """
        spectra_csv: 光谱数据 (一行一个扫描)
        clinical_csv: 临床数据 (一行一个病人)
        wave_cols: 光谱波点列
        """
        self.spectra_df = pd.read_csv(spectra_csv, sep=None, engine="python")
        self.clin_df = pd.read_csv(clinical_csv, sep=None, engine="python")
        self.wave_cols = wave_cols
        self.label_col = label_col
        self.preprocess_fn = preprocess_fn
        self.max_scans = max_scans

        # reset 临床索引，方便 merge
        if "PatientID" in self.clin_df.columns:
            self.clin_df = self.clin_df.reset_index(drop=True)
        elif self.clin_df.index.name == "PatientID":
            self.clin_df = self.clin_df.reset_index()

        # 从光谱的 Sample 提取患者ID（如 "P1000-3" -> "P1000"）
        self.spectra_df["PatientID"] = self.spectra_df["Sample"].apply(lambda s: re.match(r"(P\d+)", s).group(1))

        # 匹配共同的 PatientID
        self.samples = sorted(list(set(self.spectra_df["PatientID"]).intersection(set(self.clin_df["PatientID"]))))

        self.items = []
        for pid in self.samples:
            spec_rows = self.spectra_df[self.spectra_df["PatientID"] == pid]
            clin_row = self.clin_df[self.clin_df["PatientID"] == pid].iloc[0]

            # 光谱矩阵
            spectra = spec_rows[self.wave_cols].values
            if self.preprocess_fn:
                spectra = np.stack([self.preprocess_fn(s) for s in spectra])
            else:
                spectra = spectra.astype(np.float32)

            # 截断 / 补齐
            S, L = spectra.shape
            if S > self.max_scans:
                spectra = spectra[:self.max_scans]
            elif S < self.max_scans:
                pad_len = self.max_scans - S
                spectra = np.concatenate([spectra, np.zeros((pad_len, L), dtype=np.float32)], axis=0)

            mask = np.zeros(self.max_scans, dtype=bool)
            mask[:min(S, self.max_scans)] = True

            # 临床特征
            tabular = clin_row.drop(["PatientID", self.label_col], errors="ignore").values.astype(np.float32)
            label = 1 if clin_row[self.label_col] == "DM" else 0  # 将字符串类别映射为 int

            self.items.append({
                "spectra": spectra,
                "mask": mask,
                "tabular": tabular,
                "label": label
            })

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        item = self.items[idx]
        return {
            "spectra": torch.tensor(item["spectra"], dtype=torch.float32),
            "mask": torch.tensor(item["mask"], dtype=torch.bool),
            "tabular": torch.tensor(item["tabular"], dtype=torch.float32),
            "label": torch.tensor(item["label"], dtype=torch.long)
        }


# ----------------------------
# collate_fn
# ----------------------------
def collate_fn(batch):
    spectra = torch.stack([b["spectra"] for b in batch])   # [B, S, L]
    mask = torch.stack([b["mask"] for b in batch])         # [B, S]
    tabular = torch.stack([b["tabular"] for b in batch])   # [B, D]
    labels = torch.stack([b["label"] for b in batch])      # [B]
    return {"spectra": spectra, "mask": mask, "tabular": tabular, "label": labels}
