import json
import sys
from typing import Iterable, Optional
from pathlib import Path
import pandas as pd

# 添加项目根目录到Python路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from code.config import get_config


def read_excel_with_fallback(path: Path) -> pd.DataFrame:
	# 使用 openpyxl 引擎读取中文路径 Excel
	return pd.read_excel(path, engine='openpyxl')


def clean_dataframe(
	df: pd.DataFrame,
	max_missing_rate: float,
	allowed_labels: Optional[Iterable] = None,
	id_column: Optional[str] = None,
) -> pd.DataFrame:
	"""
	按照给定缺失率阈值清洗数据，并可选保留指定标签。
	"""
	# 确保存在至少一列作为标签
	if df.shape[1] < 2:
		raise ValueError('数据列数不足：至少需要标签列和一个特征列')

	# 复制一份，便于安全修改
	df = df.copy()

	# 若指定 ID 列但数据中不存在，则用原始索引创建
	if id_column:
		if id_column not in df.columns:
			insert_pos = 1 if df.shape[1] >= 1 else 0
			df.insert(insert_pos, id_column, df.index)

	# 去重
	df = df.drop_duplicates(ignore_index=True)

	# 删除全空列
	df = df.dropna(axis=1, how='all')

	# 计算缺失率并删除高缺失列（保留首列标签）
	label_col = df.columns[0]
	protected_cols = {label_col}
	if id_column and id_column in df.columns:
		protected_cols.add(id_column)

	features = [c for c in df.columns if c not in protected_cols]
	if features:
		missing_rate = df[features].isna().mean()
		drop_cols = [c for c, r in missing_rate.items() if r > max_missing_rate]
		if drop_cols:
			df = df.drop(columns=drop_cols)

	# 删除常量列（保留标签列）
	const_cols = []
	for c in df.columns[1:]:
		if c in protected_cols:
			continue
		non_null_unique = df[c].dropna().unique()
		if len(non_null_unique) <= 1:
			const_cols.append(c)
	if const_cols:
		df = df.drop(columns=const_cols)

	# 过滤非法标签（若提供），同时丢弃缺失标签
	if allowed_labels is not None:
		allowed_set = set(allowed_labels)
		df = df[df[label_col].isin(allowed_set)]
	else:
		df = df[df[label_col].notna()]

	# 将标签尝试转换为整数，便于后续建模
	try:
		df[label_col] = df[label_col].astype('Int64')
	except (ValueError, TypeError):
		df[label_col] = df[label_col].astype(str)

	df = df.reset_index(drop=True)

	return df


def main():
	cfg = get_config()
	print(f'[clean] 读取数据: {cfg.data_path}')
	df = read_excel_with_fallback(cfg.data_path)
	print(f'[clean] 原始样本数/特征数: {df.shape[0]} / {df.shape[1]}')

	df_clean = clean_dataframe(
		df,
		cfg.max_missing_rate,
		cfg.allowed_labels,
		getattr(cfg, 'id_column_name', None),
	)
	print(f'[clean] 清洗后样本数/特征数: {df_clean.shape[0]} / {df_clean.shape[1]}')

	out_csv = cfg.output_dir / 'cleaned_data.csv'
	df_clean.to_csv(out_csv, index=False)
	print(f'[clean] 已保存: {out_csv}')

	# 记录简单概览
	summary = {
		'n_rows': int(df_clean.shape[0]),
		'n_cols': int(df_clean.shape[1]),
		'label': df_clean.columns[0],
		'allowed_labels': list(cfg.allowed_labels),
		'id_column': getattr(cfg, 'id_column_name', None),
	}
	with open(cfg.output_dir / 'clean_summary.json', 'w', encoding='utf-8') as f:
		json.dump(summary, f, ensure_ascii=False, indent=2)
	print('[clean] 完成')


if __name__ == '__main__':
	main()
