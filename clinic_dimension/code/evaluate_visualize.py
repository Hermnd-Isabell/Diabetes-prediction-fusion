import json
import joblib
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.metrics import roc_curve, auc, precision_recall_curve, confusion_matrix, ConfusionMatrixDisplay

# 添加项目根目录到Python路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from code.config import get_config


sns.set(style='whitegrid', font='SimHei')


def plot_roc(y_true, y_proba, out_path):
	fpr, tpr, _ = roc_curve(y_true, y_proba)
	roc_auc = auc(fpr, tpr)
	plt.figure(figsize=(6, 5))
	plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.3f})')
	plt.plot([0, 1], [0, 1], color='navy', lw=1, linestyle='--')
	plt.xlim([0.0, 1.0])
	plt.ylim([0.0, 1.05])
	plt.xlabel('False Positive Rate')
	plt.ylabel('True Positive Rate')
	plt.title('ROC Curve')
	plt.legend(loc="lower right")
	plt.tight_layout()
	plt.savefig(out_path, dpi=150)
	plt.close()


def plot_pr(y_true, y_proba, out_path):
	precision, recall, _ = precision_recall_curve(y_true, y_proba)
	plt.figure(figsize=(6, 5))
	plt.plot(recall, precision, color='purple', lw=2)
	plt.xlabel('Recall')
	plt.ylabel('Precision')
	plt.title('Precision-Recall Curve')
	plt.tight_layout()
	plt.savefig(out_path, dpi=150)
	plt.close()


def plot_confusion(y_true, y_pred, out_path):
	cm = confusion_matrix(y_true, y_pred)
	disp = ConfusionMatrixDisplay(confusion_matrix=cm)
	disp.plot(cmap='Blues')
	plt.title('Confusion Matrix')
	plt.tight_layout()
	plt.savefig(out_path, dpi=150)
	plt.close()


def plot_feature_importance(pipe, out_path, top_n=30):
	# 从 One-Hot 后的模型中获取重要性
	model = pipe.named_steps['model']
	importances = model.feature_importances_

	# 获取经过 ColumnTransformer 和 OneHotEncoder 的特征名
	prep = pipe.named_steps['prep']
	feature_names = []
	for name, trans, cols in prep.transformers_:
		if name == 'num':
			feature_names.extend(cols)
		elif name == 'cat':
			ohe = trans.named_steps['onehot']
			cat_feature_names = ohe.get_feature_names_out(cols).tolist()
			feature_names.extend(cat_feature_names)

	imp_df = pd.DataFrame({'feature': feature_names, 'importance': importances})
	imp_df = imp_df.sort_values('importance', ascending=False).head(top_n)

	plt.figure(figsize=(8, max(4, top_n * 0.25)))
	sns.barplot(data=imp_df, x='importance', y='feature', palette='viridis')
	plt.title('Top Feature Importances')
	plt.tight_layout()
	plt.savefig(out_path, dpi=150)
	plt.close()


def main():
	cfg = get_config()
	pipe_path = cfg.output_dir / 'model_pipeline.joblib'
	test_csv = cfg.output_dir / 'test_set.csv'
	if not pipe_path.exists() or not test_csv.exists():
		raise FileNotFoundError('请先运行 model_train.py 以生成模型与测试集')

	pipe = joblib.load(pipe_path)
	test_df = pd.read_csv(test_csv)
	label_col = test_df.columns[-1]

	X_test = test_df.drop(columns=[label_col])
	y_test = test_df[label_col].astype(int)
	proba = pipe.predict_proba(X_test)[:, 1]
	y_pred = (proba >= 0.5).astype(int)

	plot_roc(y_test, proba, cfg.output_dir / 'roc_curve.png')
	plot_pr(y_test, proba, cfg.output_dir / 'pr_curve.png')
	plot_confusion(y_test, y_pred, cfg.output_dir / 'confusion_matrix.png')
	plot_feature_importance(pipe, cfg.output_dir / 'feature_importance.png')
	print('[eval] 已输出 ROC/PR/混淆矩阵/特征重要性 图像到 outputs/')


if __name__ == '__main__':
	main()
