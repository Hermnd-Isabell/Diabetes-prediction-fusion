import os
from typing import Dict, Any

import numpy as np
import pandas as pd


def _infer_feature_columns(df: pd.DataFrame) -> np.ndarray:
	"""
	自动识别 feature_* 列，并按列名排序后返回对应的 NumPy 数组（float32）。
	"""
	feature_cols = [c for c in df.columns if c.startswith("feature_")]
	if not feature_cols:
		raise ValueError("未在 CSV 中找到任何以 'feature_' 开头的特征列")

	# 按 feature_ 后面的数字排序，确保顺序为 feature_0, feature_1, ...
	def _feature_index(name: str) -> int:
		try:
			return int(name.split("_", 1)[1])
		except Exception:
			# 非标准命名时放在最后但仍保留
			return 10**9

	feature_cols_sorted = sorted(feature_cols, key=_feature_index)
	return df[feature_cols_sorted].astype(np.float32).values


def load_spectrum_embedding(path: str) -> Dict[str, Dict[str, Any]]:
	"""
	加载光谱模态的 embedding CSV，并按 PatientID 聚合成字典。

	输入 CSV 要求列头至少包含：
	- PatientID
	- Split
	- Label
	- feature_*

	返回：
	{
	    patient_id(str): {
	        "embedding": np.ndarray[Ds] (float32),
	        "label": int,
	        "split": str  # "train" / "val" / "test"
	    },
	    ...
	}
	"""
	if not os.path.exists(path):
		raise FileNotFoundError(f"spectrum embedding CSV 不存在: {path}")

	df = pd.read_csv(path)

	required_cols = {"PatientID", "Split", "Label"}
	missing = required_cols - set(df.columns)
	if missing:
		raise ValueError(f"spectrum embedding 缺少必要列: {missing}")

	# 识别特征矩阵
	features = _infer_feature_columns(df)

	# 确保 PatientID 为字符串
	patient_ids = df["PatientID"].astype(str).values
	labels = df["Label"].astype(int).values
	splits = df["Split"].astype(str).values

	emb_dict: Dict[str, Dict[str, Any]] = {}

	for pid, emb, lbl, split in zip(patient_ids, features, labels, splits):
		# 若同一 Patient 出现多次，目前简单覆盖为“最后一条”，
		# 如需更复杂聚合（平均、多条样本等），之后可在此扩展。
		emb_dict[pid] = {
			"embedding": np.asarray(emb, dtype=np.float32),
			"label": int(lbl),
			"split": split,
		}

	return emb_dict


def load_clinical_embedding(path: str) -> Dict[str, Dict[str, Any]]:
	"""
	加载临床模态的 embedding CSV。

	输入 CSV 要求列头至少包含：
	- PatientID
	- Label
	- feature_*

	返回：
	{
	    patient_id(str): {
	        "embedding": np.ndarray[Dc] (float32),
	        "label": int,
	    },
	    ...
	}
	"""
	if not os.path.exists(path):
		raise FileNotFoundError(f"clinical embedding CSV 不存在: {path}")

	df = pd.read_csv(path)

	required_cols = {"PatientID", "Label"}
	missing = required_cols - set(df.columns)
	if missing:
		raise ValueError(f"clinical embedding 缺少必要列: {missing}")

	features = _infer_feature_columns(df)
	patient_ids = df["PatientID"].astype(str).values
	labels = df["Label"].astype(int).values

	emb_dict: Dict[str, Dict[str, Any]] = {}
	for pid, emb, lbl in zip(patient_ids, features, labels):
		emb_dict[pid] = {
			"embedding": np.asarray(emb, dtype=np.float32),
			"label": int(lbl),
		}

	return emb_dict


def align_by_patient_id(
	spectrum_dict: Dict[str, Dict[str, Any]],
	clinical_dict: Dict[str, Dict[str, Any]],
) -> Dict[str, Dict[str, Any]]:
	"""
	按 PatientID 对齐光谱与临床 embedding。

	规则：
	- 以光谱端 patient_id 为主（因为 split 信息来自光谱）
	- 若某 patient 在临床端不存在 → clinical_embedding = None
	- 若某 patient 在光谱端不存在 → 不加入
	- 若两端 label 不一致 → 以光谱 label 为准

	返回结构：
	{
	    patient_id(str): {
	        "spectra_embedding": np.ndarray[Ds] 或 None,
	        "clinical_embedding": np.ndarray[Dc] 或 None,
	        "label": int,
	        "split": str,  # "train" / "val" / "test"
	    },
	    ...
	}
	"""
	aligned: Dict[str, Dict[str, Any]] = {}

	for pid, spec_item in spectrum_dict.items():
		spec_emb = spec_item.get("embedding", None)
		spec_label = int(spec_item.get("label"))
		spec_split = spec_item.get("split")

		clin_item = clinical_dict.get(pid)
		if clin_item is not None:
			clin_emb = clin_item.get("embedding", None)
			# label 冲突时，以光谱为准
			# clin_label = int(clin_item.get("label"))
		else:
			clin_emb = None

		aligned[pid] = {
			"spectra_embedding": None if spec_emb is None else np.asarray(spec_emb, dtype=np.float32),
			"clinical_embedding": None if clin_emb is None else np.asarray(clin_emb, dtype=np.float32),
			"label": spec_label,
			"split": spec_split,
		}

	return aligned


