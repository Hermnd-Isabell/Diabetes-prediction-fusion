import json
import sys
from pathlib import Path
from typing import Dict, List

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# 添加项目根目录到Python路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from code.config import get_config


def infer_column_types(df: pd.DataFrame, id_column: str | None = None) -> Dict[str, List[str]]:
	label_col = df.columns[0]
	drop_cols = [label_col]
	if id_column and id_column in df.columns:
		drop_cols.append(id_column)
	feature_df = df.drop(columns=drop_cols, errors='ignore')

	numeric_cols = feature_df.select_dtypes(include=['number']).columns.tolist()
	# 其余全部视为类别列
	categorical_cols = [c for c in feature_df.columns if c not in numeric_cols]

	return {
		'label': label_col,
		'numeric': numeric_cols,
		'categorical': categorical_cols,
	}


def build_preprocessor(numeric_cols: List[str], categorical_cols: List[str], min_freq: float) -> ColumnTransformer:
	numeric_pipeline = Pipeline(steps=[
		('imputer', SimpleImputer(strategy='median')),
	])
	categorical_pipeline = Pipeline(steps=[
		('imputer', SimpleImputer(strategy='most_frequent')),
		('onehot', OneHotEncoder(handle_unknown='ignore', min_frequency=min_freq, sparse_output=False)),
	])
	preprocessor = ColumnTransformer(
		transformers=[
			('num', numeric_pipeline, numeric_cols),
			('cat', categorical_pipeline, categorical_cols),
		],
		remainder='drop',
	)
	return preprocessor


def main():
	cfg = get_config()
	data_csv = cfg.output_dir / 'cleaned_data.csv'
	if not data_csv.exists():
		raise FileNotFoundError(f'未找到清洗后数据: {data_csv}，请先运行 data_cleaning.py')

	df = pd.read_csv(data_csv)
	id_column = getattr(cfg, 'id_column_name', None)
	if id_column and id_column not in df.columns:
		# 兼容早期清洗结果缺失 ID 的情况
		df.insert(1, id_column, pd.Series(range(1, len(df) + 1), name=id_column))
	elif not id_column:
		id_column = 'PatientID'
	patient_ids = df[id_column].astype(str) if id_column in df.columns else pd.Series(
		[f'P{i+1:04d}' for i in range(len(df))],
		name=id_column or 'PatientID',
		dtype='object',
	)

	types = infer_column_types(df, id_column=id_column)

	label_col = types['label']
	numeric_cols = types['numeric']
	categorical_cols = types['categorical']

	preprocessor = build_preprocessor(numeric_cols, categorical_cols, cfg.onehot_min_frequency)
	X_drop_cols = [label_col]
	if id_column and id_column in df.columns:
		X_drop_cols.append(id_column)
	X = df.drop(columns=X_drop_cols)
	y = df[label_col].astype(int)

	X_processed = preprocessor.fit_transform(X)
	feature_names = preprocessor.get_feature_names_out()

	# 标准化 + 可选 PCA
	scaler = StandardScaler()
	X_scaled = scaler.fit_transform(X_processed)

	metadata = {
		'n_rows': int(X.shape[0]),
		'n_original_features': int(X.shape[1]),
		'n_encoded_features': int(X_processed.shape[1]),
		'n_numeric_features': len(numeric_cols),
		'n_categorical_features': len(categorical_cols),
		'perform_pca': bool(cfg.perform_pca),
	}

	if cfg.perform_pca:
		if cfg.pca_n_components is not None:
			pca = PCA(n_components=cfg.pca_n_components, random_state=cfg.random_seed)
		else:
			pca = PCA(n_components=cfg.pca_explained_variance, random_state=cfg.random_seed)

		X_final = pca.fit_transform(X_scaled)
		final_feature_names = [f'pc_{i+1}' for i in range(X_final.shape[1])]
		metadata['pca_explained_variance_ratio'] = pca.explained_variance_ratio_.tolist()
		metadata['n_final_features'] = int(X_final.shape[1])
	else:
		pca = None
		X_final = X_scaled
		final_feature_names = feature_names.tolist()
		metadata['n_final_features'] = int(X_final.shape[1])

	# 合并标签并保存处理后的数据
	processed_df = pd.DataFrame(X_final, columns=final_feature_names)
	processed_df.insert(0, label_col, y.values)

	output_dir = cfg.output_dir
	output_dir.mkdir(parents=True, exist_ok=True)

	with open(output_dir / 'columns.json', 'w', encoding='utf-8') as f:
		json.dump(types, f, ensure_ascii=False, indent=2)

	processed_df.to_csv(output_dir / 'processed_data.csv', index=False)

	# 导出多模态可用的 PatientID + Label + feature_0...feature_D
	# 临床导出保持与 processed_df 使用同一特征表示，仅统一命名规范
	n_features = X_final.shape[1]
	clinical_feature_cols = [f'feature_{i}' for i in range(n_features)]

	clinical_features_only = processed_df.drop(columns=[label_col])
	clinical_df = pd.DataFrame(clinical_features_only.values, columns=clinical_feature_cols)
	clinical_df.insert(0, 'Label', y.values)
	clinical_df.insert(0, 'PatientID', patient_ids.values)
	clinical_path = getattr(cfg, 'clinical_export_path', project_root / 'data' / 'clinical.csv')
	clinical_path.parent.mkdir(parents=True, exist_ok=True)
	clinical_df.to_csv(clinical_path, index=False)
	metadata['clinical_export_path'] = str(clinical_path)
	metadata['patient_id_column'] = 'PatientID'

	# 保存元信息与转换器
	metadata['feature_names'] = list(final_feature_names)
	with open(output_dir / 'feature_engineering_summary.json', 'w', encoding='utf-8') as f:
		json.dump(metadata, f, ensure_ascii=False, indent=2)

	joblib.dump({
		'preprocessor': preprocessor,
		'scaler': scaler,
		'pca': pca,
		'feature_names_original': feature_names.tolist(),
		'final_feature_names': final_feature_names,
	}, output_dir / 'feature_pipeline.joblib')

	print('[feat] 数值特征数:', len(numeric_cols))
	print('[feat] 类别特征数:', len(categorical_cols))
	print(f'[feat] 编码后特征维度: {X_processed.shape[1]}')
	print(f"[feat] 最终特征维度: {metadata['n_final_features']}")
	if cfg.perform_pca and pca is not None:
		print('[feat] PCA 方差占比:',
		      ', '.join(f'{v:.3f}' for v in metadata['pca_explained_variance_ratio']))
	print(f'[feat] 已保存列定义: {output_dir / "columns.json"}')
	print(f'[feat] 已保存处理数据: {output_dir / "processed_data.csv"}')
	print(f'[feat] 已保存流水线: {output_dir / "feature_pipeline.joblib"}')


if __name__ == '__main__':
	main()
