import json
import joblib
import sys
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.metrics import roc_auc_score, f1_score, accuracy_score
from lightgbm import LGBMClassifier

# 添加项目根目录到Python路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from code.config import get_config


def build_preprocessor(numeric_cols, categorical_cols, min_freq):
	numeric_pipeline = Pipeline(steps=[
		('imputer', SimpleImputer(strategy='median')),
	])
	categorical_pipeline = Pipeline(steps=[
		('imputer', SimpleImputer(strategy='most_frequent', fill_value='Missing')),
		('onehot', OneHotEncoder(handle_unknown='ignore', min_frequency=min_freq, sparse_output=False))
	])
	preprocessor = ColumnTransformer(
		transformers=[
			('num', numeric_pipeline, numeric_cols),
			('cat', categorical_pipeline, categorical_cols),
		]
	)
	return preprocessor


def main():
	cfg = get_config()
	data_csv = cfg.output_dir / 'cleaned_data.csv'
	cols_json = cfg.output_dir / 'columns.json'
	if not data_csv.exists():
		raise FileNotFoundError('请先运行 data_cleaning.py')
	if not cols_json.exists():
		raise FileNotFoundError('请先运行 feature_processing.py')

	df = pd.read_csv(data_csv)
	with open(cols_json, 'r', encoding='utf-8') as f:
		cols_def = json.load(f)

	label_col = cols_def['label']
	numeric_cols = cols_def['numeric']
	categorical_cols = cols_def['categorical']

	X = df.drop(columns=[label_col])
	y = df[label_col].astype(int)

	X_train, X_test, y_train, y_test = train_test_split(
		X, y, test_size=cfg.test_size, random_state=cfg.random_seed, stratify=y
	)

	# 从训练集中再切分验证集用于早停
	X_tr, X_val, y_tr, y_val = train_test_split(
		X_train, y_train, test_size=cfg.val_size, random_state=cfg.random_seed, stratify=y_train
	)

	preprocessor = build_preprocessor(numeric_cols, categorical_cols, cfg.onehot_min_frequency)
	model = LGBMClassifier(**cfg.lgbm_params)

	pipe = Pipeline(steps=[('prep', preprocessor), ('model', model)])

	# 先拟合预处理器
	X_tr_processed = preprocessor.fit_transform(X_tr)
	X_val_processed = preprocessor.transform(X_val)
	X_test_processed = preprocessor.transform(X_test)

	# 训练模型（不使用早停，因为版本兼容性问题）
	model.fit(X_tr_processed, y_tr)

	# 更新pipeline中的模型
	pipe.named_steps['model'] = model

	# 评估测试集（支持多分类）
	proba = pipe.predict_proba(X_test)
	y_pred = pipe.predict(X_test)
	num_classes = len(cfg.allowed_labels)
	if num_classes == 2:
		# 二分类
		metrics = {
			'auc': float(roc_auc_score(y_test, proba[:, 1])),
			'f1': float(f1_score(y_test, y_pred)),
			'accuracy': float(accuracy_score(y_test, y_pred)),
		}
	else:
		# 多分类
		from sklearn.preprocessing import label_binarize
		y_test_bin = label_binarize(y_test, classes=list(cfg.allowed_labels))
		try:
			macro_auc = float(roc_auc_score(y_test_bin, proba, multi_class='ovr', average='macro'))
		except ValueError:
			macro_auc = None
		metrics = {
			'macro_auc': macro_auc,
			'f1_weighted': float(f1_score(y_test, y_pred, average='weighted')),
			'f1_macro': float(f1_score(y_test, y_pred, average='macro')),
			'accuracy': float(accuracy_score(y_test, y_pred)),
		}
	with open(cfg.output_dir / 'metrics.json', 'w', encoding='utf-8') as f:
		json.dump(metrics, f, ensure_ascii=False, indent=2)
	print('[train] 测试集指标:', metrics)

	# 保存流水线模型
	joblib.dump(pipe, cfg.output_dir / 'model_pipeline.joblib')
	print('[train] 已保存模型: outputs/model_pipeline.joblib')

	# 保存测试集，后续评估绘图使用
	test_df = X_test.copy()
	test_df[label_col] = y_test.values
	test_df.to_csv(cfg.output_dir / 'test_set.csv', index=False)
	print('[train] 已保存测试集: outputs/test_set.csv')


if __name__ == '__main__':
	main()
