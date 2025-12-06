from typing import Dict, Any, List

import numpy as np
import torch
from torch.utils.data import Dataset


class EmbeddingMultimodalDataset(Dataset):
	"""
	基于对齐后的 embedding 结果构建的多模态 Dataset。

	输入数据格式（来自 multimodal.embedding_loader.align_by_patient_id）:
	aligned = {
	    patient_id: {
	        "spectra_embedding": np.ndarray 或 None,
	        "clinical_embedding": np.ndarray 或 None,
	        "label": int,
	        "split": "train" / "val" / "test"
	    },
	    ...
	}

	本 Dataset 只做张量封装与切分，不做任何标准化或额外处理。
	"""

	def __init__(self, aligned: Dict[str, Dict[str, Any]], split: str, dropout_config: Dict[str, float] = None):
		"""
		Args:
			aligned: 按 patient 对齐的 embedding 字典
			split: 选择的划分 ("train" / "val" / "test")
			dropout_config: 模态 dropout 配置 {"spectra": float, "clinical": float}，默认两者为 0.0
		"""
		self.split = split
		self.items: List[Dict[str, Any]] = []
		
		# 模态 dropout 配置（仅在训练时使用，验证/测试集应设为 0）
		if dropout_config is None:
			dropout_config = {"spectra": 0.0, "clinical": 0.0}
		self.dropout_spec = dropout_config.get("spectra", 0.0)
		self.dropout_clin = dropout_config.get("clinical", 0.0)

		# 先在全体样本中确定各模态的 embedding 维度
		spec_dim = None
		clin_dim = None
		for entry in aligned.values():
			if spec_dim is None and entry.get("spectra_embedding") is not None:
				spec_dim = int(np.asarray(entry["spectra_embedding"], dtype=np.float32).shape[-1])
			if clin_dim is None and entry.get("clinical_embedding") is not None:
				clin_dim = int(np.asarray(entry["clinical_embedding"], dtype=np.float32).shape[-1])
			if spec_dim is not None and clin_dim is not None:
				break

		if spec_dim is None:
			raise ValueError("EmbeddingMultimodalDataset: 所有样本的 spectra_embedding 均为 None，无法确定光谱维度。")
		if clin_dim is None:
			raise ValueError("EmbeddingMultimodalDataset: 所有样本的 clinical_embedding 均为 None，无法确定临床维度。")

		for pid, entry in aligned.items():
			if entry.get("split") != split:
				continue

			spec_emb = entry.get("spectra_embedding")
			clin_emb = entry.get("clinical_embedding")
			label = int(entry.get("label"))

			# 光谱模态：缺失时用全 0 向量占位，并记录 has_spectra=False
			if spec_emb is None:
				spectra = np.zeros(spec_dim, dtype=np.float32)
				has_spectra = False
			else:
				spectra = np.asarray(spec_emb, dtype=np.float32)
				if spectra.shape[-1] != spec_dim:
					raise ValueError(f"光谱 embedding 维度不一致: 期望 {spec_dim}, 实际 {spectra.shape[-1]}")
				has_spectra = True

			# 临床模态：缺失时用全 0 向量占位，并记录 has_tabular=False
			if clin_emb is None:
				tabular = np.zeros(clin_dim, dtype=np.float32)
				has_tabular = False
			else:
				tabular = np.asarray(clin_emb, dtype=np.float32)
				if tabular.shape[-1] != clin_dim:
					raise ValueError(f"临床 embedding 维度不一致: 期望 {clin_dim}, 实际 {tabular.shape[-1]}")
				has_tabular = True

			self.items.append(
				{
					"patient": str(pid),
					"spectra": spectra,
					"tabular": tabular,
					"has_spectra": has_spectra,
					"has_tabular": has_tabular,
					"label": label,
				}
			)

	def __len__(self) -> int:
		return len(self.items)
	
	def apply_modality_dropout(self, spectra, tabular, has_spectra, has_tabular):
		"""
		应用模态 dropout：随机将某些模态的 embedding 置零，并更新 mask。
		
		Args:
			spectra: 光谱 embedding (numpy array)
			tabular: 临床 embedding (numpy array)
			has_spectra: 是否有光谱数据 (bool)
			has_tabular: 是否有临床数据 (bool)
		
		Returns:
			(spectra, tabular, has_spectra, has_tabular) 应用 dropout 后的结果
		"""
		# 光谱模态 dropout
		if self.dropout_spec > 0 and has_spectra:
			drop_mask = np.random.rand() < self.dropout_spec
			if drop_mask:
				spectra = np.zeros_like(spectra)
				has_spectra = False
		
		# 临床模态 dropout
		if self.dropout_clin > 0 and has_tabular:
			drop_mask = np.random.rand() < self.dropout_clin
			if drop_mask:
				tabular = np.zeros_like(tabular)
				has_tabular = False
		
		return spectra, tabular, has_spectra, has_tabular

	def __getitem__(self, idx: int) -> Dict[str, Any]:
		item = self.items[idx]
		
		# 应用模态 dropout（仅在训练时生效，因为验证/测试集的 dropout_config 应为 0）
		spectra, tabular, has_spectra, has_tabular = self.apply_modality_dropout(
			item["spectra"].copy(),
			item["tabular"].copy(),
			item["has_spectra"],
			item["has_tabular"]
		)
		
		return {
			"patient": item["patient"],
			"spectra": torch.tensor(spectra, dtype=torch.float32),
			"tabular": torch.tensor(tabular, dtype=torch.float32),
			"has_spectra": torch.tensor(has_spectra, dtype=torch.bool),
			"has_tabular": torch.tensor(has_tabular, dtype=torch.bool),
			"label": torch.tensor(item["label"], dtype=torch.long),
		}


def embedding_collate_fn(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
	"""
	collate 函数：将一批样本打包成批次。

	输出结构：
	{
	    "patient": List[str],
	    "spectra": Tensor[B, D_spectra],
	    "tabular": Tensor[B, D_clinical],
	    "has_spectra": Tensor[B] (bool),
	    "has_tabular": Tensor[B] (bool),
	    "label": Tensor[B]
	}
	"""
	patients: List[str] = [b["patient"] for b in batch]
	spectra = torch.stack([b["spectra"] for b in batch], dim=0)
	tabular = torch.stack([b["tabular"] for b in batch], dim=0)
	labels = torch.stack([b["label"] for b in batch], dim=0)
	has_spectra = torch.stack([b["has_spectra"] for b in batch], dim=0)
	has_tabular = torch.stack([b["has_tabular"] for b in batch], dim=0)

	return {
		"patient": patients,
		"spectra": spectra,
		"tabular": tabular,
		"has_spectra": has_spectra,
		"has_tabular": has_tabular,
		"label": labels,
	}


