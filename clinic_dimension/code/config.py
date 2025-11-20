from pathlib import Path
from typing import Optional
from dataclasses import dataclass


@dataclass(frozen=True)
class Config:
	# 路径设置
	project_root: Path = Path(__file__).resolve().parents[1]
	data_path: Path = project_root / '原始数据' / '糖尿病标签4分类.xlsx'
	output_dir: Path = project_root / 'outputs-4分类'
	clinical_export_path: Path = project_root.parent / 'data' / 'clinical.csv'

	# 随机与数据切分
	random_seed: int = 42
	test_size: float = 0.2  # 训练/测试划分
	val_size: float = 0.2   # 训练内部再划分验证集，用于早停

	# 清洗参数
	max_missing_rate: float = 0.7  # 缺失率超过该阈值的列会被删除
	id_column_name: str = 'sample_id'  # 若原始数据无此列则自动创建

	# 标签设置
	allowed_labels: tuple = (0, 1, 2, 3)

	# 特征工程
	perform_pca: bool = True
	pca_n_components: Optional[int] = None  # 若指定则优先使用
	pca_explained_variance: float = 0.95    # 未指定组件数时保留的累计方差

	# 类别编码参数
	onehot_min_frequency: float = 0.01  # 稀有类别聚合阈值

	# 模型参数（可根据数据规模微调）
	lgbm_params: dict = None


def get_config() -> Config:
	cfg = Config()
	# 创建输出目录
	cfg.output_dir.mkdir(parents=True, exist_ok=True)
	cfg.clinical_export_path.parent.mkdir(parents=True, exist_ok=True)

	# 默认 LightGBM 参数
	default_lgbm = {
		'n_estimators': 5000,
		'learning_rate': 0.02,
		'max_depth': -1,
		'num_leaves': 64,
		'min_child_samples': 20,
		'subsample': 0.9,
		'colsample_bytree': 0.8,
		'reg_alpha': 0.1,
		'reg_lambda': 0.1,
		'objective': 'multiclass',
		'metric': ['multi_logloss', 'multi_error'],
		'num_class': len(cfg.allowed_labels),
		'class_weight': 'balanced',
		'n_jobs': -1,
		'random_state': cfg.random_seed,
	}
	object.__setattr__(cfg, 'lgbm_params', default_lgbm)
	return cfg
