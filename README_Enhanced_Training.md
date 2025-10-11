# 🚀 增强版多模态模型训练系统

一个功能强大的多模态深度学习训练系统，支持四个先进的模型架构，并提供丰富的可视化和可解释性分析功能。

## 📊 支持的模型

### 1. **AttentionMultimodal** - 注意力机制模型
- 跨模态注意力融合
- 自注意力增强
- 多头注意力机制

### 2. **ConcatFusion** - 基线融合模型
- 简单特征拼接
- 全连接层分类
- 轻量级架构

### 3. **MMTMMultimodal** - 多模态张量融合
- 张量融合机制
- 模态间交互建模
- 高效的特征融合

### 4. **TFTMultimodal** - 时序融合Transformer
- Transformer架构
- 时序建模能力
- 长距离依赖捕获

## 🎯 核心功能

### 📈 丰富的可视化
- **训练过程可视化**: 损失曲线、准确率曲线、AUC曲线
- **评估可视化**: ROC曲线、混淆矩阵、精确率-召回率曲线
- **特征可视化**: t-SNE降维、PCA分析、特征分布
- **注意力可视化**: 注意力权重热图、注意力模式分析

### 🔍 可解释性分析
- **SHAP分析**: 特征重要性解释
- **注意力分析**: 注意力权重可视化
- **特征重要性**: PCA主成分分析
- **模型对比**: 多模型性能比较

### 🏆 高级训练功能
- **早停机制**: 防止过拟合
- **学习率调度**: 自适应学习率调整
- **梯度裁剪**: 训练稳定性
- **模型保存**: 自动保存最佳模型

## 🚀 快速开始

### 1. 环境准备

```bash
# 安装依赖
pip install torch torchvision
pip install numpy pandas matplotlib seaborn
pip install scikit-learn tqdm pyyaml
pip install shap  # 用于可解释性分析
```

### 2. 数据准备

确保数据文件位于正确位置：
```
data/
├── spectra.csv      # 光谱数据
└── clinical.csv     # 临床数据
```

### 3. 运行实验

#### 方式一：使用快速启动脚本（推荐）

```bash
python run_experiments.py
```

然后选择实验选项：
- 训练单个模型
- 训练所有模型并比较
- 仅评估已训练模型

#### 方式二：使用命令行

```bash
# 训练单个模型
python enhanced_main.py --config configs/enhanced_config.yaml --model AttentionMultimodal

# 训练所有模型并比较
python enhanced_main.py --config configs/enhanced_config.yaml --train-all

# 仅评估模型
python enhanced_main.py --config configs/enhanced_config.yaml --eval-only AttentionMultimodal
```

## 📋 配置文件说明

### 主要配置项

```yaml
# 数据配置
data:
  spectra_csv: data/spectra.csv
  clinical_csv: data/clinical.csv
  train_ratio: 0.7
  val_ratio: 0.15
  test_ratio: 0.15

# 模型配置
model:
  name: AttentionMultimodal
  num_classes: 2
  spec_emb: 256
  tab_emb: 128

# 训练配置
train:
  batch_size: 8
  lr: 0.001
  epochs: 50
  early_stopping_patience: 10

# 可视化配置
visualization:
  enable: true
  plot_training_curves: true
  plot_roc_curves: true

# 可解释性配置
interpretability:
  enable: true
  shap_analysis: true
  attention_analysis: true
```

## 📊 输出结果

### 文件结构
```
results/
├── AttentionMultimodal/
│   ├── best_model.pt              # 最佳模型权重
│   ├── training_curves.png        # 训练曲线
│   ├── evaluation_plots.png       # 评估图表
│   ├── shap_analysis.png          # SHAP分析
│   ├── attention_analysis.png     # 注意力分析
│   └── results.json               # 详细结果
├── ConcatFusion/
├── MMTMMultimodal/
├── TFTMultimodal/
└── comparison/
    ├── model_comparison.png       # 模型比较图表
    ├── roc_comparison.png         # ROC曲线比较
    └── comparison_summary.json    # 比较摘要
```

### 可视化图表说明

1. **训练曲线** (`training_curves.png`)
   - 训练/验证损失曲线
   - 训练/验证准确率曲线
   - 训练/验证AUC曲线
   - 训练/验证F1分数曲线

2. **评估图表** (`evaluation_plots.png`)
   - ROC曲线
   - 精确率-召回率曲线
   - 混淆矩阵
   - 预测概率分布
   - 特征空间可视化(t-SNE)
   - 注意力权重可视化

3. **可解释性分析**
   - **SHAP分析**: 特征重要性解释
   - **注意力分析**: 注意力权重模式
   - **PCA分析**: 主成分分析

4. **模型比较**
   - 性能指标对比表
   - ROC曲线比较
   - 综合性能雷达图

## 🔧 高级用法

### 自定义模型配置

```python
# 修改配置文件中的模型参数
model:
  name: EnhancedMMTM
  num_attention_heads: 8
  fusion_strategy: hierarchical
  enable_uncertainty: true
```

### 批量实验

```python
# 在配置文件中指定要训练的模型列表
models_to_train:
  - AttentionMultimodal
  - ConcatFusion
  - MMTMMultimodal
  - TFTMultimodal
```

### 自定义可视化

```python
# 在配置文件中调整可视化选项
visualization:
  enable: true
  plot_training_curves: true
  plot_roc_curves: true
  plot_confusion_matrix: true
  plot_feature_visualization: true
```

## 📈 性能指标

系统会自动计算并展示以下指标：

- **准确率 (Accuracy)**: 整体分类准确率
- **AUC**: ROC曲线下面积
- **F1分数**: 精确率和召回率的调和平均
- **敏感性@90%特异性**: 在90%特异性下的敏感性

## 🐛 故障排除

### 常见问题

1. **CUDA内存不足**
   ```yaml
   train:
     batch_size: 4  # 减小批次大小
   ```

2. **训练速度慢**
   ```yaml
   train:
     epochs: 20  # 减少训练轮数
   ```

3. **可视化失败**
   ```yaml
   visualization:
     enable: false  # 禁用可视化
   ```

### 日志查看

训练过程中的详细日志会保存在 `results/` 目录下，包括：
- 训练进度
- 验证指标
- 错误信息
- 模型摘要

## 🤝 贡献

欢迎提交Issue和Pull Request来改进这个训练系统！

## 📄 许可证

本项目采用MIT许可证。

---

**🎉 开始你的多模态深度学习之旅吧！**

