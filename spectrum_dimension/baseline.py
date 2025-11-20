import os
import sys
import torch
from torch import nn, optim
from torch.utils.data import Dataset, DataLoader, random_split
from torch.nn import functional as F
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score, confusion_matrix, classification_report
from tqdm import tqdm

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")
os.makedirs(DEFAULT_OUTPUT_DIR, exist_ok=True)



def clean_and_filter_data(df, label_col='label', exclude_cols=None):
    """
    清理数据：过滤非数值列、处理缺失值、排除指定列
    :param df: 原始DataFrame
    :param label_col: 标签列名
    :param exclude_cols: 需要排除的列名列表（如文件名、样本ID等）
    :return: 清理后的特征X和标签y（y可为None，如果是预测模式）
    """

    df_clean = df.copy()


    if exclude_cols is None:
        exclude_cols = []

    if label_col in exclude_cols and label_col in df_clean.columns:
        exclude_cols.remove(label_col)


    for col in exclude_cols:
        if col in df_clean.columns:
            df_clean = df_clean.drop(col, axis=1)
            print(f"已排除列: {col}")


    y = None
    if label_col in df_clean.columns:
        y = df_clean[label_col]
        X = df_clean.drop(label_col, axis=1)
        print(f"标签列 '{label_col}' 已分离")
    else:
        X = df_clean
        print("未检测到标签列，按预测模式处理")


    numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()
    non_numeric_cols = [col for col in X.columns if col not in numeric_cols]

    if non_numeric_cols:
        print(f"检测到非数值列（自动过滤）: {non_numeric_cols}")
        X = X[numeric_cols]

    if len(X.columns) == 0:
        raise ValueError("过滤后无有效数值特征列！请检查数据格式")


    missing_stats = X.isnull().sum()
    missing_cols = missing_stats[missing_stats > 0].index.tolist()
    if missing_cols:
        print(f"检测到缺失值列: {missing_cols}，使用均值填充")
        for col in missing_cols:
            X[col] = X[col].fillna(X[col].mean())


    X = X.replace([np.inf, -np.inf], np.nan)
    X = X.fillna(X.mean())

    print(f"数据清理完成：{len(X)} 个样本, {len(X.columns)} 个数值特征")
    return X, y



class DiabetesDataset(Dataset):
    def __init__(self, features, labels=None, seq_len=5, img_size=(97, 97), normalize=True, scaler=None):
        """
        糖尿病二分类数据集
        :param features: 特征数据（numpy数组或DataFrame）
        :param labels: 标签数据（numpy数组，0=无糖尿病，1=有糖尿病），预测时可为None
        :param seq_len: 序列长度
        :param img_size: 生成的图像大小
        :param normalize: 是否归一化
        :param scaler: 预定义的归一化器（用于测试集/预测集）
        """
        self.features = features.values if isinstance(features, pd.DataFrame) else features
        self.labels = labels.values if isinstance(labels, pd.Series) else labels
        self.seq_len = seq_len
        self.img_size = img_size
        self.is_predict = labels is None


        if normalize:
            if scaler is None:
                self.scaler = MinMaxScaler()
                self.features = self.scaler.fit_transform(self.features)
            else:
                self.scaler = scaler
                self.features = self.scaler.transform(self.features)
        else:
            self.scaler = None

        self.feature_dim = self.features.shape[1]
        self.feature_names = features.columns.tolist() if isinstance(features, pd.DataFrame) else [f"feat_{i}" for i in
                                                                                                   range(
                                                                                                       self.feature_dim)]


        if self.feature_dim < seq_len:
            raise ValueError(f"特征数量({self.feature_dim})少于序列长度({seq_len})，请减小seq_len")

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        sample_features = self.features[idx]


        step = max(1, self.feature_dim // self.seq_len)
        seq_features = sample_features[::step][:self.seq_len]


        seq = []
        for feature in seq_features:
            img = np.ones(self.img_size, dtype=np.float32) * feature
            img = self.add_spatial_pattern(img, feature)
            seq.append(img[np.newaxis, :, :])  # [C, H, W]

        if self.is_predict:
            return torch.tensor(np.array(seq), dtype=torch.float32)
        else:
            label = torch.tensor(self.labels[idx], dtype=torch.long)
            return torch.tensor(np.array(seq), dtype=torch.float32), label

    def add_spatial_pattern(self, img, feature_value):
        """为光谱/数值特征添加空间模式，增强特征区分度"""
        h, w = img.shape
        center = (h // 2, w // 2)

        radius = int(min(h, w) * 0.3 * max(0.1, feature_value * 5)) + 1
        for i in range(h):
            for j in range(w):
                dist = np.sqrt((i - center[0]) ** 2 + (j - center[1]) ** 2)
                if dist < radius:
                    img[i, j] *= (1 + 0.7 * (1 - dist / radius))
        return img



class Residual(nn.Module):
    def __init__(self, input_channels, num_channels, use_1x1conv=False, strides=1):
        super().__init__()
        self.conv1 = nn.Conv2d(input_channels, num_channels,
                               kernel_size=3, padding=1, stride=strides)
        self.conv2 = nn.Conv2d(num_channels, num_channels,
                               kernel_size=3, padding=1)
        if use_1x1conv:
            self.conv3 = nn.Conv2d(input_channels, num_channels,
                                   kernel_size=1, stride=strides)
        else:
            self.conv3 = None
        self.bn1 = nn.BatchNorm2d(num_channels)
        self.bn2 = nn.BatchNorm2d(num_channels)

    def forward(self, X):
        Y = F.relu(self.bn1(self.conv1(X)))
        Y = self.bn2(self.conv2(Y))
        if self.conv3:
            X = self.conv3(X)
        Y += X
        return F.relu(Y)


def resnet_block(input_channels, num_channels, num_residuals, first_block=False):
    blk = []
    for i in range(num_residuals):
        if i == 0 and not first_block:
            blk.append(Residual(input_channels, num_channels,
                                use_1x1conv=True, strides=2))
        else:
            blk.append(Residual(num_channels, num_channels))
    return blk


class ResNetFeatureExtractor(nn.Module):
    def __init__(self):
        super().__init__()
        self.b1 = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm2d(64), nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )
        self.b2 = nn.Sequential(*resnet_block(64, 64, 2, first_block=True))
        self.b3 = nn.Sequential(*resnet_block(64, 128, 2))
        self.b4 = nn.Sequential(*resnet_block(128, 256, 2))
        self.b5 = nn.Sequential(*resnet_block(256, 512, 2))
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.flatten = nn.Flatten()

    def forward(self, X):
        X = self.b1(X)
        X = self.b2(X)
        X = self.b3(X)
        X = self.b4(X)
        X = self.b5(X)
        X = self.pool(X)
        return self.flatten(X)



class SelfAttentionBlock(nn.Module):
    def __init__(self, d_model, nhead=8, dim_feedforward=2048, dropout=0.3):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.attention_weights = None

    def forward(self, src):
        src2, attn_weights = self.self_attn(src, src, src)
        self.attention_weights = attn_weights
        src = src + self.dropout1(src2)
        src = self.norm1(src)


        src2 = self.linear1(src)
        src2 = F.relu(src2)
        src2 = self.dropout2(src2)
        src2 = self.linear2(src2)

        src = src + self.dropout2(src2)
        src = self.norm2(src)
        return src


class DiabetesClassifier(nn.Module):
    def __init__(self, feature_dim, seq_len=5, d_model=512, nhead=8, num_layers=2, dropout=0.3):
        super().__init__()
        self.d_model = d_model
        self.seq_len = seq_len


        self.resnet = ResNetFeatureExtractor()


        self.projection = nn.Linear(512, d_model) if 512 != d_model else nn.Identity()


        self.position_encoding = nn.Parameter(torch.randn(1, seq_len, d_model))


        self.attention_layers = nn.ModuleList([
            SelfAttentionBlock(d_model, nhead, 2048, dropout)
            for _ in range(num_layers)
        ])


        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(d_model, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 2)
        )


        self.final_attention_weights = None

    def forward(self, X):
        batch_size, seq_len, C, H, W = X.shape
        X_flat = X.view(batch_size * seq_len, C, H, W)


        resnet_feat = self.resnet(X_flat)
        feat_seq = resnet_feat.view(batch_size, seq_len, -1)


        feat_seq = self.projection(feat_seq) + self.position_encoding


        attn_feat = feat_seq
        for i, layer in enumerate(self.attention_layers):
            attn_feat = layer(attn_feat)
            if i == len(self.attention_layers) - 1:
                self.final_attention_weights = layer.attention_weights


        pooled_feat = attn_feat.mean(dim=1)

        pooled_feat = self.classifier[0](pooled_feat)
        pooled_feat = self.classifier[1](pooled_feat)
        pooled_feat = self.classifier[2](pooled_feat)
        pooled_feat = self.classifier[3](pooled_feat)
        logits = self.classifier[4](pooled_feat)

        return logits


def train_model(model, train_loader, val_loader, criterion, optimizer, device, epochs=50, patience=10,
                output_dir=DEFAULT_OUTPUT_DIR):
    """训练模型，包含早停机制"""
    model.to(device)
    best_val_acc = 0.0
    best_val_auc = 0.0
    patience_counter = 0
    train_losses, val_losses = [], []
    train_accs, val_accs = [], []

    for epoch in range(epochs):

        model.train()
        train_loss = 0.0
        train_preds, train_labels = [], []

        for batch in tqdm(train_loader, desc=f"Epoch {epoch + 1}/{epochs} (Train)"):
            inputs, labels = batch
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * inputs.size(0)
            train_preds.extend(torch.argmax(outputs, dim=1).cpu().numpy())
            train_labels.extend(labels.cpu().numpy())


        train_loss /= len(train_loader.dataset)
        train_acc = accuracy_score(train_labels, train_preds)
        train_losses.append(train_loss)
        train_accs.append(train_acc)


        model.eval()
        val_loss = 0.0
        val_preds, val_probs, val_labels = [], [], []

        with torch.no_grad():
            for batch in val_loader:
                inputs, labels = batch
                inputs, labels = inputs.to(device), labels.to(device)

                outputs = model(inputs)
                loss = criterion(outputs, labels)
                val_loss += loss.item() * inputs.size(0)

                probs = F.softmax(outputs, dim=1)[:, 1].cpu().numpy()
                preds = torch.argmax(outputs, dim=1).cpu().numpy()
                val_probs.extend(probs)
                val_preds.extend(preds)
                val_labels.extend(labels.cpu().numpy())


        val_loss /= len(val_loader.dataset)
        val_acc = accuracy_score(val_labels, val_preds)
        val_auc = roc_auc_score(val_labels, val_probs) if len(np.unique(val_labels)) > 1 else 0.0
        val_losses.append(val_loss)
        val_accs.append(val_acc)


        print(f"\nEpoch {epoch + 1} Summary:")
        print(f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f}")
        print(f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f} | Val AUC: {val_auc:.4f}")


        if val_acc > best_val_acc or (val_acc == best_val_acc and val_auc > best_val_auc):
            best_val_acc = val_acc
            best_val_auc = val_auc
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_val_acc': best_val_acc,
                'best_val_auc': best_val_auc,
            }, os.path.join(output_dir, 'best_diabetes_classifier.pth'))
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"\n早停触发！最佳验证准确率: {best_val_acc:.4f}, 最佳AUC: {best_val_auc:.4f}")
                break


    plot_training_curves(train_losses, val_losses, train_accs, val_accs, output_dir=output_dir)
    return model


def plot_training_curves(train_losses, val_losses, train_accs, val_accs, output_dir=DEFAULT_OUTPUT_DIR):
    """绘制训练损失和准确率曲线"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))


    ax1.plot(train_losses, label='Train Loss', linewidth=2)
    ax1.plot(val_losses, label='Val Loss', linewidth=2)
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title('Training & Validation Loss')
    ax1.legend()
    ax1.grid(alpha=0.3)

    ax2.plot(train_accs, label='Train Acc', linewidth=2)
    ax2.plot(val_accs, label='Val Acc', linewidth=2)
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy')
    ax2.set_title('Training & Validation Accuracy')
    ax2.legend()
    ax2.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'training_curves.png'), dpi=300)
    plt.close()


def evaluate_model(model, test_loader, device, output_dir=DEFAULT_OUTPUT_DIR):
    """评估模型在测试集上的性能"""
    model.to(device)
    model.eval()

    test_preds, test_probs, test_labels = [], [], []

    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Evaluating Test Set"):
            inputs, labels = batch
            inputs, labels = inputs.to(device), labels.to(device)

            outputs = model(inputs)
            probs = F.softmax(outputs, dim=1)[:, 1].cpu().numpy()
            preds = torch.argmax(outputs, dim=1).cpu().numpy()

            test_probs.extend(probs)
            test_preds.extend(preds)
            test_labels.extend(labels.cpu().numpy())


    test_acc = accuracy_score(test_labels, test_preds)
    test_auc = roc_auc_score(test_labels, test_probs)
    conf_matrix = confusion_matrix(test_labels, test_preds)
    class_report = classification_report(test_labels, test_preds, target_names=['无糖尿病', '有糖尿病'])


    print("\n" + "=" * 50)
    print("测试集性能评估")
    print("=" * 50)
    print(f"准确率 (Accuracy): {test_acc:.4f}")
    print(f"AUC: {test_auc:.4f}")
    print("\n混淆矩阵:")
    print(conf_matrix)
    print("\n分类报告:")
    print(class_report)


    plot_confusion_matrix(conf_matrix, output_dir=output_dir)
    return test_acc, test_auc


def plot_confusion_matrix(conf_matrix, output_dir=DEFAULT_OUTPUT_DIR):
    """绘制混淆矩阵热图"""
    plt.figure(figsize=(8, 6))
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues',
                xticklabels=['无糖尿病', '有糖尿病'],
                yticklabels=['无糖尿病', '有糖尿病'])
    plt.xlabel('预测标签')
    plt.ylabel('真实标签')
    plt.title('糖尿病分类混淆矩阵')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'confusion_matrix.png'), dpi=300)
    plt.close()


def predict_new_data(csv_path, model_path, seq_len=5, device='cuda', exclude_cols=None,
                     output_dir=DEFAULT_OUTPUT_DIR):
    """预测新数据（无标签CSV）"""
    # 加载数据
    os.makedirs(output_dir, exist_ok=True)
    data = pd.read_csv(csv_path)
    print(f"加载预测数据: {len(data)} 个样本, {data.shape[1]} 个列")


    X, _ = clean_and_filter_data(data, label_col='label', exclude_cols=exclude_cols)


    checkpoint = torch.load(model_path, map_location=device)


    model = DiabetesClassifier(feature_dim=X.shape[1], seq_len=seq_len)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()


    try:

        train_data_path = os.path.join(output_dir, 'train_data.csv')
        train_data = pd.read_csv(train_data_path)
        X_train, _ = clean_and_filter_data(train_data, label_col='label', exclude_cols=exclude_cols)
        scaler = MinMaxScaler()
        scaler.fit(X_train)
    except Exception as e:
        print(f"警告：无法加载训练集数据，使用预测数据的统计量进行归一化 - {e}")
        scaler = MinMaxScaler()
        scaler.fit(X)


    predict_dataset = DiabetesDataset(
        features=X,
        labels=None,
        seq_len=seq_len,
        normalize=True,
        scaler=scaler
    )
    predict_loader = DataLoader(predict_dataset, batch_size=16, shuffle=False)


    predictions = []
    probabilities = []

    with torch.no_grad():
        for batch in tqdm(predict_loader, desc="Predicting"):
            inputs = batch.to(device)
            outputs = model(inputs)
            probs = F.softmax(outputs, dim=1).cpu().numpy()  # [batch, 2]
            preds = torch.argmax(outputs, dim=1).cpu().numpy()

            predictions.extend(preds)
            probabilities.extend(probs)


    results = pd.DataFrame({
        '样本索引': range(len(X)),
        '预测结果': ['有糖尿病' if p == 1 else '无糖尿病' for p in predictions],
        '无糖尿病概率': [prob[0] for prob in probabilities],
        '有糖尿病概率': [prob[1] for prob in probabilities]
    })


    if 'Sample' in data.columns:
        results.insert(1, '样本ID', data['Sample'].values[:len(results)])
    elif 'sample_id' in data.columns:
        results.insert(1, '样本ID', data['sample_id'].values[:len(results)])


    prediction_path = os.path.join(output_dir, 'prediction_results.csv')
    results.to_csv(prediction_path, index=False, encoding='utf-8-sig')
    print(f"\n预测结果已保存到 {prediction_path}")
    print("\n预测结果统计:")
    print(results['预测结果'].value_counts())
    print("\n前10个样本预测结果:")
    print(results.head(10))

    return results



def run_diabetes_classification(csv_path, seq_len=5, epochs=50, batch_size=16, lr=1e-4, exclude_cols=None,
                                output_dir=DEFAULT_OUTPUT_DIR):
    """
    运行糖尿病二分类完整流程
    :param csv_path: 输入CSV文件路径（需包含'label'列，0=无糖尿病，1=有糖尿病）
    :param seq_len: 序列长度
    :param epochs: 训练轮数
    :param batch_size: 批次大小
    :param lr: 学习率
    :param exclude_cols: 需要排除的列名列表（如文件名、样本ID等）
    """

    os.makedirs(output_dir, exist_ok=True)
    print("=" * 50)
    print("加载数据...")
    data = pd.read_csv(csv_path)
    print(f"原始数据概况: {len(data)} 个样本, {data.shape[1]} 个列")


    X, y = clean_and_filter_data(data, label_col='label', exclude_cols=exclude_cols)

    if y is not None:
        unique_labels = np.unique(y)
        if len(unique_labels) != 2:
            raise ValueError(f"标签必须是二分类（仅包含0和1），当前标签: {unique_labels}")
        y = y.astype(int)
        print(f"类别分布: \n{y.value_counts()}")

        X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)
        X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.67, random_state=42,
                                                        stratify=y_temp)

        train_data = pd.concat([X_train, y_train], axis=1)
        train_data_path = os.path.join(output_dir, 'train_data.csv')
        train_data.to_csv(train_data_path, index=False)
        print(f"\n数据集划分完成:")
        print(f"训练集: {len(X_train)} 样本, 验证集: {len(X_val)} 样本, 测试集: {len(X_test)} 样本")


        train_dataset = DiabetesDataset(X_train, y_train, seq_len=seq_len)
        val_dataset = DiabetesDataset(X_val, y_val, seq_len=seq_len, scaler=train_dataset.scaler)
        test_dataset = DiabetesDataset(X_test, y_test, seq_len=seq_len, scaler=train_dataset.scaler)

        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=2)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=2)
    else:
        raise ValueError("训练模式需要标签列 'label'")


    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n使用设备: {device}")

    model = DiabetesClassifier(
        feature_dim=X.shape[1],
        seq_len=seq_len,
        d_model=512,
        nhead=8,
        num_layers=2,
        dropout=0.3
    )

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-5)


    print("\n" + "=" * 50)
    print("开始训练模型...")
    model = train_model(model, train_loader, val_loader, criterion, optimizer, device, epochs=epochs,
                        output_dir=output_dir)


    print("\n" + "=" * 50)
    print("加载最佳模型进行测试集评估...")
    checkpoint_path = os.path.join(output_dir, 'best_diabetes_classifier.pth')
    best_checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(best_checkpoint['model_state_dict'])
    test_acc, test_auc = evaluate_model(model, test_loader, device, output_dir=output_dir)


    plot_attention_weights(model, next(iter(test_loader))[0].to(device), sample_idx=0, output_dir=output_dir)

    print("\n" + "=" * 50)
    print("糖尿病二分类任务完成！")
    print(f"最佳模型已保存为: {checkpoint_path}")
    print(f"训练曲线: {os.path.join(output_dir, 'training_curves.png')}")
    print(f"混淆矩阵: {os.path.join(output_dir, 'confusion_matrix.png')}")
    print(f"注意力权重图: {os.path.join(output_dir, 'attention_weights.png')}")
    print(f"测试集准确率: {test_acc:.4f}, AUC: {test_auc:.4f}")


def plot_attention_weights(model, inputs, sample_idx=0, output_dir=DEFAULT_OUTPUT_DIR):
    """可视化自注意力权重矩阵"""
    model.eval()
    with torch.no_grad():
        model(inputs)

    attn_weights = model.final_attention_weights  # [batch, nhead, seq_len, seq_len]
    seq_len = attn_weights.shape[2]
    avg_attn = attn_weights[sample_idx].mean(dim=0).cpu().numpy()

    plt.figure(figsize=(10, 8))
    sns.heatmap(avg_attn, cmap="viridis", annot=True, fmt=".2f",
                xticklabels=[f"特征组 {i + 1}" for i in range(seq_len)],
                yticklabels=[f"特征组 {i + 1}" for i in range(seq_len)])
    plt.title("自注意力权重矩阵（特征组间依赖关系）", fontsize=15)
    plt.xlabel("Key 特征组", fontsize=12)
    plt.ylabel("Query 特征组", fontsize=12)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "attention_weights.png"), dpi=300)
    plt.close()
    print("注意力权重图已保存为: attention_weights.png")


if __name__ == "__main__":

    CSV_PATH = os.path.join("data", "spectra.csv")
    SEQ_LEN = 5
    EPOCHS = 50
    BATCH_SIZE = 16
    LEARNING_RATE = 1e-4
    EXCLUDE_COLS = ['Sample', 'Group', 'filename']
    OUTPUT_DIR = DEFAULT_OUTPUT_DIR


    MODE = "train"

    if MODE == "train":

        run_diabetes_classification(
            csv_path=CSV_PATH,
            seq_len=SEQ_LEN,
            epochs=EPOCHS,
            batch_size=BATCH_SIZE,
            lr=LEARNING_RATE,
            exclude_cols=EXCLUDE_COLS,
            output_dir=OUTPUT_DIR
        )
    elif MODE == "predict":

        PREDICT_CSV = "new_diabetes_data.csv"
        MODEL_PATH = os.path.join(OUTPUT_DIR, "best_diabetes_classifier.pth")
        predict_new_data(
            csv_path=PREDICT_CSV,
            model_path=MODEL_PATH,
            seq_len=SEQ_LEN,
            exclude_cols=EXCLUDE_COLS,
            output_dir=OUTPUT_DIR
        )