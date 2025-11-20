# 糖尿病并发症二分类（LightGBM 全流程）

本项目实现了从原始数据清洗、特征处理、模型训练到可视化评估的完整流水线，适配数据维度高、缺失值多、类别丰富的特征场景。模型采用 LightGBM 进行二分类：0 表示无并发症，1 表示有并发症（神经或血管）。

## 目录结构
- `code/config.py`: 全局配置（路径、参数、随机种子等）
- `code/data_cleaning.py`: 读取 Excel 原始数据并进行基础清洗
- `code/feature_processing.py`: 自动识别特征类型，导出列清单
- `code/model_train.py`: 训练 LightGBM，保存预处理器、模型与测试集
- `code/evaluate_visualize.py`: 评估指标与可视化（ROC/PR/混淆矩阵/特征重要性）
- `code/run_all.py`: 串联一键运行
- `outputs/`: 运行产物（清洗数据、列清单、模型、指标、图表等）

## 环境准备
```bash
pip install -r requirements.txt
```

## 数据准备
请将原始数据文件放在：`原始数据/糖尿病标签2分类.xlsx`
- Excel 第一列为分类标签（0/1）

## 一键运行
```bash
python code/run_all.py
```

运行完成后，主要输出位于 `outputs/`：
- `cleaned_data.csv`: 清洗后的数据（保留首列标签以及 `sample_id`）
- `columns.json`: 数值/类别特征列清单
- `model_pipeline.joblib`: 预处理+模型的推理流水线
- `test_set.csv`: 测试集（含标签与特征）
- `metrics.json`: 指标（AUC/F1/Accuracy 等）
- `roc_curve.png`, `pr_curve.png`, `confusion_matrix.png`, `feature_importance.png`

## 复现要点
- 清洗阶段：
  - 删除全空列、缺失率超阈值列（默认 0.8）
  - 删除常量列与重复行
- 特征阶段：
  - 数值列使用中位数填充；类别列填充为 "Missing" 并做 One-Hot，`handle_unknown='ignore'`
  - One-Hot 采用 `min_frequency=0.01` 自动聚合稀有类别
- 训练阶段：
  - 使用分层划分训练/测试，并在训练内部再切分验证集以启用早停
  - `class_weight='balanced'` 以缓解潜在类别不均衡

## 预测使用
训练完成后，可在任何脚本中加载 `model_pipeline.joblib` 对原始格式数据直接预测：
```python
import joblib
import pandas as pd

pipe = joblib.load('outputs/model_pipeline.joblib')
X = pd.read_csv('outputs/cleaned_data.csv')  # 或者准备与训练时相同列名的数据
X = X.drop(columns=[X.columns[0]])  # 确保去掉标签列
proba = pipe.predict_proba(X)[:, 1]
```

如需自定义参数，可修改 `code/config.py`。
