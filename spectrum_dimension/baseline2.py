import torch
from torch import nn, optim
from torch.utils.data import Dataset, DataLoader, Subset
from torch.nn import functional as F
from torch.utils.tensorboard import SummaryWriter
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_auc_score,
    roc_curve, auc
)
from sklearn.utils.class_weight import compute_class_weight
from tqdm import tqdm
import os
import logging
from collections import defaultdict, OrderedDict
import json
import warnings
from datetime import datetime
from typing import Dict, List, Tuple, Optional, Union

warnings.filterwarnings('ignore')



class Config:
    """配置类：集中管理所有参数，避免硬编码"""

    CSV_PATH = "diabetes_raman_merged_updated.csv"
    LABEL_COL = "标签"
    EXCLUDE_COLS = ['Sample', 'Group']
    CLASS_NAMES = ["类别0", "类别1", "类别2", "类别3"]
    PATIENT_COL = "Sample"

    MODEL_NAMES = ["MLP_Baseline", "Feature_Interaction_MLP", "Light_CNN_MLP"]
    HIDDEN_DIM = 128
    DROPOUT = 0.3
    NUM_CLASSES = 4
    INTERACTION_TOP_K = 30

    BATCH_SIZE = 32
    EPOCHS = 100
    LEARNING_RATE = 1e-4
    WEIGHT_DECAY = 1e-5
    PATIENCE = 15
    MIN_DELTA = 1e-4
    USE_CLASS_WEIGHT = True
    VALIDATION_SPLIT = 0.15
    TEST_SPLIT = 0.15


    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    USE_MULTI_GPU = False
    SEED = 42

    OUTPUT_DIR = f"diabetes_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    MODEL_DIR = os.path.join(OUTPUT_DIR, "models")
    FIGURE_DIR = os.path.join(OUTPUT_DIR, "figures")
    LOG_DIR = os.path.join(OUTPUT_DIR, "logs")
    TENSORBOARD_DIR = os.path.join(OUTPUT_DIR, "tensorboard")

    @classmethod
    def init_directories(cls):
        """初始化输出目录"""
        for dir_path in [cls.OUTPUT_DIR, cls.MODEL_DIR, cls.FIGURE_DIR, cls.LOG_DIR, cls.TENSORBOARD_DIR]:
            os.makedirs(dir_path, exist_ok=True)

config = Config()
config.init_directories()


def setup_logger():
    """配置日志：同时输出到控制台和文件"""
    logger = logging.getLogger("DiabetesClassification")
    logger.setLevel(logging.INFO)

    if logger.handlers:
        return logger

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    ))

    file_handler = logging.FileHandler(
        os.path.join(config.LOG_DIR, "training.log"),
        encoding='utf-8'
    )
    file_handler.setFormatter(logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    ))

    logger.addHandler(console_handler)
    logger.addHandler(file_handler)
    return logger


logger = setup_logger()

def set_seed(seed: int = config.SEED):
    """设置随机种子，保证实验可复现"""
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


set_seed()


def calculate_cnn_output_dim(input_dim: int, kernel_size: int = 3, padding: int = 1, stride: int = 2) -> int:

    out_dim = (input_dim + 2 * padding - kernel_size) // stride + 1
    return out_dim



def clean_data(
        df: pd.DataFrame,
        label_col: str,
        exclude_cols: Optional[List[str]] = None,
        logger: logging.Logger = logger
) -> Tuple[pd.DataFrame, pd.Series]:
    """
    增强版数据清理：更全面的异常处理和数据校验
    """
    df_clean = df.copy()
    logger.info(f"原始数据形状: {df_clean.shape}")

    if exclude_cols is None:
        exclude_cols = []
    exclude_cols = [col for col in exclude_cols if col in df_clean.columns and col != label_col]
    if exclude_cols:
        df_clean = df_clean.drop(exclude_cols, axis=1)
        logger.info(f"已排除非特征列: {exclude_cols}")

    if label_col not in df_clean.columns:
        raise ValueError(f"数据中未找到标签列 '{label_col}'")

    y = df_clean[label_col]
    X = df_clean.drop(columns=[label_col])

    numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()
    non_numeric_cols = [col for col in X.columns if col not in numeric_cols]
    if non_numeric_cols:
        logger.warning(f"自动过滤非数值特征列: {non_numeric_cols}")
        X = X[numeric_cols]

    if len(X.columns) == 0:
        raise ValueError("过滤后无有效数值特征，请检查数据格式")
    if len(X) == 0:
        raise ValueError("数据为空，请检查输入数据")

    missing_stats = X.isnull().sum()
    missing_cols = missing_stats[missing_stats > 0].index.tolist()
    if missing_cols:
        missing_ratio = (missing_stats[missing_cols] / len(X) * 100).round(2)
        logger.warning(f"缺失值统计（比例）: {dict(zip(missing_cols, missing_ratio))}")
        drop_cols = [col for col, ratio in zip(missing_cols, missing_ratio) if ratio > 50]
        fill_cols = [col for col in missing_cols if col not in drop_cols]

        if drop_cols:
            X = X.drop(drop_cols, axis=1)
            logger.warning(f"删除缺失比例>50%的列: {drop_cols}")
        if fill_cols:
            for col in fill_cols:
                X[col] = X[col].fillna(X[col].mean())
            logger.info(f"用均值填充缺失值列: {fill_cols}")

    X = X.replace([np.inf, -np.inf], np.nan).fillna(X.mean())
    for col in X.columns:
        mean = X[col].mean()
        std = X[col].std()
        upper_bound = mean + 3 * std
        lower_bound = mean - 3 * std
        outlier_count = ((X[col] < lower_bound) | (X[col] > upper_bound)).sum()
        if outlier_count > 0:
            logger.warning(f"特征 '{col}' 检测到 {outlier_count} 个极端值（3σ原则），已用边界值替换")
            X[col] = X[col].clip(lower=lower_bound, upper=upper_bound)


    try:
        y = y.astype(int)
    except:
        logger.info("标签列非整数类型，进行标签编码")
        le = LabelEncoder()
        y = pd.Series(le.fit_transform(y), index=y.index)
        logger.info(f"标签映射: {dict(zip(le.classes_, range(len(le.classes_))))}")


    unique_labels = np.unique(y)
    label_counts = y.value_counts().sort_index()
    logger.info(f"数据清理完成: {len(X)} 样本 × {len(X.columns)} 特征")
    logger.info(f"标签分布: {dict(label_counts)}")
    logger.info(f"标签类别: {sorted(unique_labels.tolist())}")


    if len(unique_labels) != config.NUM_CLASSES:
        logger.warning(f"实际标签类别数 ({len(unique_labels)}) 与配置的类别数 ({config.NUM_CLASSES}) 不匹配")

    return X, y


class DiabetesDataset(Dataset):

    def __init__(
            self,
            X: pd.DataFrame,
            y: pd.Series,
            scaler: Optional[StandardScaler] = None,
            normalize: bool = True,
            feature_names: Optional[List[str]] = None
    ):
        self.features = X.values.astype(np.float32)
        self.labels = y.values.astype(np.int64)
        self.feature_names = feature_names if feature_names is not None else X.columns.tolist()
        self.feature_dim = self.features.shape[1]


        self.normalize = normalize
        if normalize:
            self.scaler = scaler if scaler is not None else StandardScaler()
            self.features = self.scaler.fit_transform(self.features) if scaler is None else self.scaler.transform(
                self.features)
        else:
            self.scaler = None

        self.indices = np.arange(len(self.features))

    def __len__(self) -> int:
        return len(self.features)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, int]:
        feature = torch.tensor(self.features[idx], dtype=torch.float32)
        label = torch.tensor(self.labels[idx], dtype=torch.long)
        return feature, label, self.indices[idx]

    def get_class_weights(self) -> torch.Tensor:
        """计算类别权重（用于处理类别不平衡）"""
        class_weights = compute_class_weight(
            class_weight='balanced',
            classes=np.unique(self.labels),
            y=self.labels
        )
        return torch.tensor(class_weights, dtype=torch.float32)


def split_data_by_patient(
        X: pd.DataFrame,
        y: pd.Series,
        patient_col: str,
        raw_df: pd.DataFrame,
        train_ratio: float = 0.7,
        val_ratio: float = 0.15,
        logger: logging.Logger = logger
) -> Tuple[DiabetesDataset, DiabetesDataset, DiabetesDataset]:

    if patient_col not in raw_df.columns:
        logger.warning(f"未找到患者列 '{patient_col}'，使用普通分层划分")
        X_train_val, X_test, y_train_val, y_test = train_test_split(
            X, y, test_size=config.TEST_SPLIT, random_state=config.SEED, stratify=y
        )
        X_train, X_val, y_train, y_val = train_test_split(
            X_train_val, y_train_val, test_size=config.VALIDATION_SPLIT / (1 - config.TEST_SPLIT),
            random_state=config.SEED, stratify=y_train_val
        )
    else:

        patients = raw_df[patient_col].unique()
        np.random.shuffle(patients)

        train_size = int(train_ratio * len(patients))
        val_size = int(val_ratio * len(patients))

        train_patients = patients[:train_size]
        val_patients = patients[train_size:train_size + val_size]
        test_patients = patients[train_size + val_size:]

        logger.info(
            f"患者划分: 训练集 {len(train_patients)} 人, 验证集 {len(val_patients)} 人, 测试集 {len(test_patients)} 人")

        train_mask = raw_df[patient_col].isin(train_patients)
        val_mask = raw_df[patient_col].isin(val_patients)
        test_mask = raw_df[patient_col].isin(test_patients)

        X_train, y_train = X[train_mask], y[train_mask]
        X_val, y_val = X[val_mask], y[val_mask]
        X_test, y_test = X[test_mask], y[test_mask]


    train_dataset = DiabetesDataset(X_train, y_train)
    val_dataset = DiabetesDataset(X_val, y_val, scaler=train_dataset.scaler)
    test_dataset = DiabetesDataset(X_test, y_test, scaler=train_dataset.scaler)

    logger.info(f"最终数据划分:")
    logger.info(f"  训练集: {len(train_dataset)} 样本 (特征维度: {train_dataset.feature_dim})")
    logger.info(f"  验证集: {len(val_dataset)} 样本")
    logger.info(f"  测试集: {len(test_dataset)} 样本")

    for name, dataset in [("训练集", train_dataset), ("验证集", val_dataset), ("测试集", test_dataset)]:
        label_counts = pd.Series(dataset.labels).value_counts().sort_index()
        logger.info(f"  {name}标签分布: {dict(label_counts)}")

    return train_dataset, val_dataset, test_dataset


class WeightInitializer:
    """模型权重初始化工具类"""

    @staticmethod
    def init_weights(m: nn.Module):

        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Conv1d):
            nn.init.kaiming_uniform_(m.weight, mode='fan_out', nonlinearity='relu')
            if m.bias is not None:
                nn.init.zeros_(m.bias)


class MLPBaseline(nn.Module):


    def __init__(
            self,
            input_dim: int,
            num_classes: int = config.NUM_CLASSES,
            hidden_dim: int = config.HIDDEN_DIM,
            dropout: float = config.DROPOUT
    ):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_dim),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_dim // 2),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, num_classes)
        )

        self.apply(WeightInitializer.init_weights)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)


class FeatureInteractionMLP(nn.Module):


    def __init__(
            self,
            input_dim: int,
            num_classes: int = config.NUM_CLASSES,
            hidden_dim: int = config.HIDDEN_DIM,
            dropout: float = config.DROPOUT,
            top_k: int = config.INTERACTION_TOP_K
    ):
        super().__init__()
        self.input_dim = input_dim
        self.top_k = min(top_k, input_dim)
        self.linear1 = nn.Linear(input_dim, hidden_dim)
        self.bn1 = nn.BatchNorm1d(hidden_dim)


        self.interaction_dim = self.top_k * (self.top_k - 1) // 2
        self.interaction = nn.Linear(self.interaction_dim, hidden_dim // 2)
        self.bn2 = nn.BatchNorm1d(hidden_dim // 2)


        self.output = nn.Sequential(
            nn.Linear(hidden_dim + hidden_dim // 2, hidden_dim // 2),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_dim // 2),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, num_classes)
        )


        self.apply(WeightInitializer.init_weights)
        logger.info(f"特征交互维度: {self.interaction_dim} (Top {self.top_k}特征)")

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        linear_feat = F.relu(self.bn1(self.linear1(x)))

        top_k_feat = x[:, :self.top_k]
        interaction_pairs = torch.bmm(
            top_k_feat.unsqueeze(2),
            top_k_feat.unsqueeze(1)
        )

        mask = torch.triu(torch.ones(self.top_k, self.top_k, device=x.device), diagonal=1)
        interaction_feat = interaction_pairs.masked_select(mask.bool()).view(x.size(0), -1)


        interaction_feat = F.relu(self.bn2(self.interaction(interaction_feat)))


        combined_feat = torch.cat([linear_feat, interaction_feat], dim=1)
        return self.output(combined_feat)


class LightCNNMLP(nn.Module):


    def __init__(
            self,
            input_dim: int,
            num_classes: int = config.NUM_CLASSES,
            hidden_dim: int = config.HIDDEN_DIM,
            dropout: float = config.DROPOUT
    ):
        super().__init__()
        self.input_dim = input_dim


        self.cnn = nn.Sequential(
            nn.Conv1d(1, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm1d(16),
            nn.MaxPool1d(2, stride=2),
            nn.Conv1d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm1d(32),
            nn.MaxPool1d(2, stride=2)
        )


        cnn_dim1 = calculate_cnn_output_dim(input_dim)
        cnn_dim2 = calculate_cnn_output_dim(cnn_dim1)
        self.cnn_output_dim = cnn_dim2 * 32
        logger.info(f"CNN输出维度: {self.cnn_output_dim} (输入维度: {input_dim})")


        self.mlp = nn.Sequential(
            nn.Linear(self.cnn_output_dim, hidden_dim),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_dim),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes)
        )
        self.apply(WeightInitializer.init_weights)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.unsqueeze(1)
        cnn_feat = self.cnn(x).flatten(1)
        return self.mlp(cnn_feat)


def create_model(model_name: str, input_dim: int) -> nn.Module:

    model_dict = {
        "MLP_Baseline": MLPBaseline,
        "Feature_Interaction_MLP": FeatureInteractionMLP,
        "Light_CNN_MLP": LightCNNMLP
    }

    if model_name not in model_dict:
        raise ValueError(f"不支持的模型名称: {model_name}，可选: {list(model_dict.keys())}")

    model = model_dict[model_name](input_dim=input_dim)


    if config.USE_MULTI_GPU and torch.cuda.device_count() > 1:
        logger.info(f"使用 {torch.cuda.device_count()} 个GPU训练")
        model = nn.DataParallel(model)

    return model



class EarlyStopping:


    def __init__(
            self,
            patience: int = config.PATIENCE,
            min_delta: float = config.MIN_DELTA,
            mode: str = "max",
            logger: logging.Logger = logger
    ):
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.logger = logger

        self.counter = 0
        self.best_score = None
        self.early_stop = False

    def __call__(self, current_score: float) -> bool:

        if self.best_score is None:
            self.best_score = current_score
            return False


        if self.mode == "max":
            improvement = current_score - self.best_score
        else:
            improvement = self.best_score - current_score

        if improvement > self.min_delta:
            self.best_score = current_score
            self.counter = 0
            return False
        else:
            self.counter += 1
            self.logger.warning(
                f"早停计数器: {self.counter}/{self.patience} (当前分数: {current_score:.4f}, 最佳分数: {self.best_score:.4f})")
            if self.counter >= self.patience:
                self.logger.info(f"触发早停！")
                self.early_stop = True
                return True

        return False



def train_multiclass_model(
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        input_dim: int,
        model_name: str,
        class_names: List[str] = config.CLASS_NAMES,
        logger: logging.Logger = logger
) -> Tuple[Dict[str, List[float]], str]:

    model.to(config.DEVICE)


    if config.USE_CLASS_WEIGHT:
        class_weights = train_loader.dataset.get_class_weights().to(config.DEVICE)
        logger.info(f"使用类别权重: {class_weights.cpu().numpy()}")
        criterion = nn.CrossEntropyLoss(weight=class_weights)
    else:
        criterion = nn.CrossEntropyLoss()

    optimizer = optim.AdamW(
        model.parameters(),
        lr=config.LEARNING_RATE,
        weight_decay=config.WEIGHT_DECAY
    )

    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=10, T_mult=2, eta_min=1e-6
    )

    early_stopping = EarlyStopping(patience=config.PATIENCE, mode="max", logger=logger)

    writer = SummaryWriter(os.path.join(config.TENSORBOARD_DIR, model_name))

    history = defaultdict(list)
    best_val_f1 = 0.0
    best_model_path = os.path.join(config.MODEL_DIR, f"best_{model_name}.pth")

    logger.info(f"\n{'=' * 50} 开始训练 {model_name} {'=' * 50}")
    logger.info(f"设备: {config.DEVICE}")
    logger.info(f"训练轮次: {config.EPOCHS}")
    logger.info(f"批次大小: {config.BATCH_SIZE}")
    logger.info(f"初始学习率: {config.LEARNING_RATE}")

    for epoch in range(config.EPOCHS):
        model.train()
        train_loss = 0.0
        train_preds = []
        train_labels = []

        for features, labels, _ in tqdm(train_loader, desc=f"Epoch {epoch + 1}/{config.EPOCHS} (Train)"):
            features, labels = features.to(config.DEVICE), labels.to(config.DEVICE)

            optimizer.zero_grad()
            outputs = model(features)
            loss = criterion(outputs, labels)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)  # 梯度裁剪防止梯度爆炸
            optimizer.step()

            train_loss += loss.item() * features.size(0)
            _, preds = torch.max(outputs, 1)
            train_preds.extend(preds.cpu().numpy())
            train_labels.extend(labels.cpu().numpy())


        train_loss /= len(train_loader.dataset)
        train_acc = accuracy_score(train_labels, train_preds)
        train_f1 = f1_score(train_labels, train_preds, average='weighted')


        model.eval()
        val_loss = 0.0
        val_preds = []
        val_labels = []
        val_probs = []

        with torch.no_grad():
            for features, labels, _ in val_loader:
                features, labels = features.to(config.DEVICE), labels.to(config.DEVICE)
                outputs = model(features)
                loss = criterion(outputs, labels)

                val_loss += loss.item() * features.size(0)
                probs = F.softmax(outputs, dim=1)
                _, preds = torch.max(outputs, 1)

                val_preds.extend(preds.cpu().numpy())
                val_labels.extend(labels.cpu().numpy())
                val_probs.extend(probs.cpu().numpy())


        val_loss /= len(val_loader.dataset)
        val_acc = accuracy_score(val_labels, val_preds)
        val_f1 = f1_score(val_labels, val_preds, average='weighted')


        scheduler.step()
        current_lr = optimizer.param_groups[0]['lr']


        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['train_acc'].append(train_acc)
        history['val_acc'].append(val_acc)
        history['train_f1'].append(train_f1)
        history['val_f1'].append(val_f1)
        history['lr'].append(current_lr)


        writer.add_scalars('Loss', {'train': train_loss, 'val': val_loss}, epoch)
        writer.add_scalars('Accuracy', {'train': train_acc, 'val': val_acc}, epoch)
        writer.add_scalars('F1-Score', {'train': train_f1, 'val': val_f1}, epoch)
        writer.add_scalar('Learning Rate', current_lr, epoch)


        if val_f1 > best_val_f1 + config.MIN_DELTA:
            best_val_f1 = val_f1
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict() if not config.USE_MULTI_GPU else model.module.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scaler': train_loader.dataset.scaler,
                'best_val_f1': best_val_f1,
                'best_val_acc': val_acc,
                'feature_names': train_loader.dataset.feature_names,
                'class_names': class_names,
                'num_classes': config.NUM_CLASSES,
                'input_dim': input_dim,
                'config': config.__dict__
            }, best_model_path)
            logger.info(f"保存最佳模型 (Val F1: {best_val_f1:.4f})")


        logger.info(
            f"Epoch {epoch + 1:3d} | LR: {current_lr:.6f} | "
            f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f} | Train F1: {train_f1:.4f} | "
            f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f} | Val F1: {val_f1:.4f} | "
            f"Best Val F1: {best_val_f1:.4f}"
        )


        if early_stopping(val_f1):
            break

    writer.close()
    logger.info(f"\n训练完成！最佳模型路径: {best_model_path}")

    return history, best_model_path



def calculate_multiclass_auc(
        y_true: np.ndarray,
        y_probs: np.ndarray,
        num_classes: int
) -> Tuple[float, Dict[int, float]]:

    try:
        # 宏观AUC
        macro_auc = roc_auc_score(y_true, y_probs, multi_class='ovr', average='macro')
        # 每个类别的AUC
        class_auc = {}
        for i in range(num_classes):
            y_true_binary = (y_true == i).astype(int)
            y_prob_binary = y_probs[:, i]
            fpr, tpr, _ = roc_curve(y_true_binary, y_prob_binary)
            class_auc[i] = auc(fpr, tpr)
        return macro_auc, class_auc
    except:
        logger.warning("AUC计算失败（可能是类别样本不足）")
        return 0.0, {i: 0.0 for i in range(num_classes)}


def evaluate_multiclass_model(
        model: nn.Module,
        test_loader: DataLoader,
        model_name: str,
        class_names: List[str] = config.CLASS_NAMES,
        logger: logging.Logger = logger
) -> Dict[str, Union[float, List[float], np.ndarray]]:

    model.to(config.DEVICE)
    model.eval()

    all_preds = []
    all_labels = []
    all_probs = []
    all_indices = []

    logger.info(f"\n{'=' * 50} 评估 {model_name} {'=' * 50}")

    with torch.no_grad():
        for features, labels, indices in tqdm(test_loader, desc=f"Evaluating {model_name}"):
            features, labels = features.to(config.DEVICE), labels.to(config.DEVICE)
            outputs = model(features)
            probs = F.softmax(outputs, dim=1)
            _, preds = torch.max(outputs, 1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())
            all_indices.extend(indices.numpy())


    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    all_probs = np.array(all_probs)
    all_indices = np.array(all_indices)


    acc = accuracy_score(all_labels, all_preds)
    precision = precision_score(all_labels, all_preds, average='weighted', zero_division=0)
    recall = recall_score(all_labels, all_preds, average='weighted', zero_division=0)
    f1 = f1_score(all_labels, all_preds, average='weighted', zero_division=0)

    class_precision = precision_score(all_labels, all_preds, average=None, zero_division=0)
    class_recall = recall_score(all_labels, all_preds, average=None, zero_division=0)
    class_f1 = f1_score(all_labels, all_preds, average=None, zero_division=0)

    cm = confusion_matrix(all_labels, all_preds)
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] * 100  # 按行归一化（百分比）

    macro_auc, class_auc = calculate_multiclass_auc(all_labels, all_probs, config.NUM_CLASSES)


    logger.info(f"\n{model_name} 评估结果汇总:")
    logger.info(f"准确率 (Accuracy): {acc:.4f}")
    logger.info(f"加权精确率 (Precision): {precision:.4f}")
    logger.info(f"加权召回率 (Recall): {recall:.4f}")
    logger.info(f"加权F1分数: {f1:.4f}")
    logger.info(f"宏观AUC-ROC: {macro_auc:.4f}")

    logger.info(f"\n每个类别的详细指标:")
    for i, class_name in enumerate(class_names):
        logger.info(f"{class_name} (标签{i}):")
        logger.info(
            f"  精确率: {class_precision[i]:.4f} | 召回率: {class_recall[i]:.4f} | F1分数: {class_f1[i]:.4f} | AUC: {class_auc[i]:.4f}")

    logger.info(f"\n混淆矩阵（原始）:")
    logger.info("        " + " ".join([f"{cn:^8}" for cn in class_names]))
    for i, row in enumerate(cm):
        logger.info(f"{class_names[i]:<8}" + " ".join([f"{val:^8d}" for val in row]))

    logger.info(f"\n混淆矩阵（归一化，%）:")
    logger.info("        " + " ".join([f"{cn:^8}" for cn in class_names]))
    for i, row in enumerate(cm_normalized):
        logger.info(f"{class_names[i]:<8}" + " ".join([f"{val:^8.1f}" for val in row]))

    logger.info(f"\n分类报告:")
    logger.info(classification_report(
        all_labels, all_preds, target_names=class_names, digits=4, zero_division=0
    ))

    return {
        'acc': acc, 'precision': precision, 'recall': recall, 'f1': f1, 'macro_auc': macro_auc,
        'class_precision': class_precision.tolist(),
        'class_recall': class_recall.tolist(),
        'class_f1': class_f1.tolist(),
        'class_auc': class_auc,
        'confusion_matrix': cm.tolist(),
        'confusion_matrix_normalized': cm_normalized.tolist(),
        'probs': all_probs.tolist(),
        'preds': all_preds.tolist(),
        'labels': all_labels.tolist(),
        'indices': all_indices.tolist(),
        'class_names': class_names
    }



def plot_multiclass_results(
        history: Dict[str, List[float]],
        eval_results: Dict[str, Union[float, List[float]]],
        model_name: str,
        output_dir: str = config.FIGURE_DIR,
        logger: logging.Logger = logger
):

    os.makedirs(output_dir, exist_ok=True)
    class_names = eval_results['class_names']
    num_classes = len(class_names)

    plt.style.use('seaborn-v0_8-whitegrid')
    sns.set_palette('Set2')
    fig, axes = plt.subplots(3, 2, figsize=(18, 20))
    plt.suptitle(f"{model_name} 糖尿病四分类结果分析", fontsize=20, y=0.98)

    ax1 = axes[0, 0]
    color1 = 'tab:blue'
    ax1.plot(history['train_loss'], label='训练损失', color=color1, linewidth=2)
    ax1.plot(history['val_loss'], label='验证损失', color=color1, linestyle='--', linewidth=2)
    ax1.set_xlabel('训练轮次', fontsize=12)
    ax1.set_ylabel('损失值', fontsize=12, color=color1)
    ax1.tick_params(axis='y', labelcolor=color1)
    ax1.set_title('训练/验证损失曲线', fontsize=14, fontweight='bold')
    ax1.legend(loc='upper right')
    ax1.grid(alpha=0.3)

    ax1_twin = ax1.twinx()
    color2 = 'tab:red'
    ax1_twin.plot(history['lr'], label='学习率', color=color2, linewidth=2, alpha=0.7)
    ax1_twin.set_ylabel('学习率', fontsize=12, color=color2)
    ax1_twin.tick_params(axis='y', labelcolor=color2)
    ax1_twin.legend(loc='lower right')

    ax2 = axes[0, 1]
    ax2.plot(history['train_acc'], label='训练准确率', color='green', linewidth=2)
    ax2.plot(history['val_acc'], label='验证准确率', color='green', linestyle='--', linewidth=2)
    ax2.plot(history['train_f1'], label='训练F1分数', color='orange', linewidth=2)
    ax2.plot(history['val_f1'], label='验证F1分数', color='orange', linestyle='--', linewidth=2)
    ax2.set_xlabel('训练轮次', fontsize=12)
    ax2.set_ylabel('分数', fontsize=12)
    ax2.set_title('准确率和F1分数曲线', fontsize=14, fontweight='bold')
    ax2.legend()
    ax2.grid(alpha=0.3)


    ax3 = axes[1, 0]
    cm_normalized = np.array(eval_results['confusion_matrix_normalized'])
    sns.heatmap(
        cm_normalized, annot=True, fmt='.1f', cmap='YlOrRd', ax=ax3,
        xticklabels=class_names, yticklabels=class_names,
        cbar_kws={'label': '百分比 (%)'}
    )
    ax3.set_xlabel('预测标签', fontsize=12)
    ax3.set_ylabel('真实标签', fontsize=12)
    ax3.set_title('归一化混淆矩阵', fontsize=14, fontweight='bold')


    ax4 = axes[1, 1]
    metrics = ['精确率', '召回率', 'F1分数', 'AUC']
    x = np.arange(num_classes)
    width = 0.2

    class_precision = eval_results['class_precision']
    class_recall = eval_results['class_recall']
    class_f1 = eval_results['class_f1']
    class_auc = [eval_results['class_auc'][i] for i in range(num_classes)]

    ax4.bar(x - 1.5 * width, class_precision, width, label='精确率', alpha=0.8)
    ax4.bar(x - 0.5 * width, class_recall, width, label='召回率', alpha=0.8)
    ax4.bar(x + 0.5 * width, class_f1, width, label='F1分数', alpha=0.8)
    ax4.bar(x + 1.5 * width, class_auc, width, label='AUC', alpha=0.8)

    ax4.set_xlabel('类别', fontsize=12)
    ax4.set_ylabel('分数', fontsize=12)
    ax4.set_title('各类别性能指标对比', fontsize=14, fontweight='bold')
    ax4.set_xticks(x)
    ax4.set_xticklabels(class_names)
    ax4.legend()
    ax4.grid(alpha=0.3, axis='y')
    ax4.set_ylim(0, 1.05)


    ax5 = axes[2, 0]
    y_true = np.array(eval_results['labels'])
    y_probs = np.array(eval_results['probs'])

    for i, class_name in enumerate(class_names):
        y_true_binary = (y_true == i).astype(int)
        y_prob_binary = y_probs[:, i]
        fpr, tpr, _ = roc_curve(y_true_binary, y_prob_binary)
        roc_auc = auc(fpr, tpr)
        ax5.plot(fpr, tpr, label=f'{class_name} (AUC = {roc_auc:.3f})', linewidth=2)

    ax5.plot([0, 1], [0, 1], 'k--', linewidth=1, label='随机猜测')
    ax5.set_xlabel('假阳性率 (FPR)', fontsize=12)
    ax5.set_ylabel('真阳性率 (TPR)', fontsize=12)
    ax5.set_title(f'多分类AUC-ROC曲线 (宏观AUC = {eval_results["macro_auc"]:.3f})', fontsize=14, fontweight='bold')
    ax5.legend(loc='lower right')
    ax5.grid(alpha=0.3)


    ax6 = axes[2, 1]
    sample_indices = np.argsort(np.array(eval_results['labels']))[:12]
    x = np.arange(num_classes)
    width = 0.15

    for i, idx in enumerate(sample_indices):
        probs = y_probs[idx]
        true_label = y_true[idx]
        pred_label = np.array(eval_results['preds'])[idx]
        color = 'green' if true_label == pred_label else 'red'

        ax6.bar(
            x + (i - 5.5) * width, probs, width,
            alpha=0.6, color=color,
            label=f"样本{idx + 1}" if i < 5 else ""
        )

    ax6.set_xlabel('类别', fontsize=12)
    ax6.set_ylabel('预测概率', fontsize=12)
    ax6.set_title('前12个样本的预测概率分布（绿=正确，红=错误）', fontsize=14, fontweight='bold')
    ax6.set_xticks(x)
    ax6.set_xticklabels(class_names)
    ax6.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax6.grid(alpha=0.3, axis='y')


    plt.tight_layout(rect=[0, 0, 1, 0.97])
    save_path = os.path.join(output_dir, f"{model_name}_results.png")
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    logger.info(f"结果可视化图已保存: {save_path}")


def predict_new_samples(
        csv_path: str,
        model_path: str,
        exclude_cols: Optional[List[str]] = None,
        output_csv: str = None,
        logger: logging.Logger = logger
) -> pd.DataFrame:

    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"预测数据文件不存在: {csv_path}")
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"模型文件不存在: {model_path}")

    logger.info(f"加载模型: {model_path}")
    checkpoint = torch.load(model_path, map_location=config.DEVICE)

    feature_names = checkpoint['feature_names']
    scaler = checkpoint['scaler']
    class_names = checkpoint['class_names']
    num_classes = checkpoint['num_classes']
    input_dim = checkpoint['input_dim']
    model_config = checkpoint.get('config', {})

    logger.info(f"模型配置: 特征数={input_dim}, 类别数={num_classes}")
    logger.info(f"类别名称: {class_names}")

    logger.info(f"加载预测数据: {csv_path}")
    new_data = pd.read_csv(csv_path)
    logger.info(f"预测数据形状: {new_data.shape}")

    dummy_df = new_data.copy()
    dummy_df['dummy_label'] = 0
    X_new, _ = clean_data(dummy_df, label_col='dummy_label', exclude_cols=exclude_cols, logger=logger)

    missing_feats = [feat for feat in feature_names if feat not in X_new.columns]
    if missing_feats:
        raise ValueError(f"新数据缺少必要特征: {missing_feats}")

    extra_feats = [feat for feat in X_new.columns if feat not in feature_names]
    if extra_feats:
        logger.warning(f"新数据包含额外特征（将被忽略）: {extra_feats[:10]}...")  # 只显示前10个

    X_new = X_new[feature_names]

    predict_dataset = DiabetesDataset(
        X_new, pd.Series([0] * len(X_new)),
        scaler=scaler, normalize=True,
        feature_names=feature_names
    )
    predict_loader = DataLoader(predict_dataset, batch_size=32, shuffle=False)

    model = create_model(
        model_name=os.path.basename(model_path).split('_')[1],
        input_dim=input_dim
    )

    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(config.DEVICE)
    model.eval()

    logger.info("开始预测新样本...")
    all_preds = []
    all_probs = []
    all_sample_indices = []

    with torch.no_grad():
        for features, _, indices in tqdm(predict_loader, desc="Predicting"):
            features = features.to(config.DEVICE)
            outputs = model(features)
            probs = F.softmax(outputs, dim=1)
            _, preds = torch.max(outputs, 1)

            all_preds.extend(preds.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())
            all_sample_indices.extend(indices.numpy())

    results = new_data.copy()
    results['预测标签'] = all_preds
    results['预测类别'] = [class_names[p] for p in all_preds]
    results['预测置信度'] = [max(probs) for probs in all_probs]

    for i, class_name in enumerate(class_names):
        results[f'{class_name}_概率'] = [round(probs[i], 4) for probs in all_probs]

    if output_csv is None:
        output_csv = os.path.join(config.OUTPUT_DIR,
                                  f"prediction_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv")

    results.to_csv(output_csv, index=False, encoding='utf-8-sig')
    logger.info(f"预测结果已保存: {output_csv}")

    logger.info(f"\n预测统计:")
    pred_count = results['预测类别'].value_counts()
    for class_name, count in pred_count.items():
        logger.info(f"  {class_name}: {count} 个样本 ({count / len(results) * 100:.1f}%)")

    logger.info(f"\n预测置信度统计:")
    logger.info(f"  平均置信度: {results['预测置信度'].mean():.4f}")
    logger.info(f"  置信度标准差: {results['预测置信度'].std():.4f}")
    logger.info(f"  最高置信度: {results['预测置信度'].max():.4f}")
    logger.info(f"  最低置信度: {results['预测置信度'].min():.4f}")

    logger.info(f"\n前5个样本预测结果:")
    display_cols = ['预测标签', '预测类别', '预测置信度'] + [f'{class_name}_概率' for class_name in class_names]
    logger.info(results[display_cols].head().to_string(index=False))

    return results

def create_model_comparison_table(
        all_results: Dict[str, Dict[str, Union[float, List[float]]]],
        output_csv: str = None,
        logger: logging.Logger = logger
) -> pd.DataFrame:
    if output_csv is None:
        output_csv = os.path.join(config.OUTPUT_DIR, "model_comparison.csv")

    summary_data = []
    for model_name, eval_res in all_results.items():
        row = OrderedDict()
        row['模型名称'] = model_name
        row['准确率'] = round(eval_res['acc'], 4)
        row['精确率'] = round(eval_res['precision'], 4)
        row['召回率'] = round(eval_res['recall'], 4)
        row['F1分数'] = round(eval_res['f1'], 4)
        row['宏观AUC'] = round(eval_res['macro_auc'], 4)


        for i, class_name in enumerate(eval_res['class_names']):
            row[f'{class_name}_F1'] = round(eval_res['class_f1'][i], 4)
            row[f'{class_name}_AUC'] = round(eval_res['class_auc'][i], 4)

        summary_data.append(row)

    summary_df = pd.DataFrame(summary_data)
    summary_df = summary_df.sort_values('F1分数', ascending=False).reset_index(drop=True)
    logger.info(f"\n{'=' * 80} 糖尿病四分类模型对比汇总 {'=' * 80}")
    logger.info(summary_df.to_string(index=False, max_colwidth=15))
    summary_df.to_csv(output_csv, index=False, encoding='utf-8-sig')
    logger.info(f"\n模型对比结果已保存: {output_csv}")

    best_model_name = summary_df.iloc[0]['模型名称']
    best_f1 = summary_df.iloc[0]['F1分数']
    best_auc = summary_df.iloc[0]['宏观AUC']
    logger.info(f"\n最佳模型: {best_model_name}")
    logger.info(f"  最佳F1分数: {best_f1:.4f}")
    logger.info(f"  最佳宏观AUC: {best_auc:.4f}")

    return summary_df, best_model_name


def main():
    """整合流程"""
    logger.info(f"\n{'=' * 100} 糖尿病Raman光谱四分类模型训练开始 {'=' * 100}")
    logger.info(f"开始时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info(
        f"配置参数: {json.dumps({k: v for k, v in config.__dict__.items() if not k.startswith('_')}, ensure_ascii=False, indent=2)}")

    try:

        logger.info(f"\n{'=' * 50} 数据处理 {'=' * 50}")
        raw_data = pd.read_csv(config.CSV_PATH)
        logger.info(f"成功加载数据: {config.CSV_PATH}")

        X_clean, y_clean = clean_data(
            raw_data,
            label_col=config.LABEL_COL,
            exclude_cols=config.EXCLUDE_COLS,
            logger=logger
        )


        train_dataset, val_dataset, test_dataset = split_data_by_patient(
            X_clean, y_clean,
            patient_col=config.PATIENT_COL,
            raw_df=raw_data,
            logger=logger
        )


        train_loader = DataLoader(train_dataset, batch_size=config.BATCH_SIZE, shuffle=True, num_workers=4)
        val_loader = DataLoader(val_dataset, batch_size=config.BATCH_SIZE, shuffle=False, num_workers=4)
        test_loader = DataLoader(test_dataset, batch_size=config.BATCH_SIZE, shuffle=False, num_workers=4)

        all_results = {}
        all_histories = {}
        all_model_paths = {}

        for model_name in config.MODEL_NAMES:

            model = create_model(model_name, input_dim=train_dataset.feature_dim)
            logger.info(f"\n模型结构: {model_name}")
            logger.info(f"模型参数量: {sum(p.numel() for p in model.parameters()):,}")


            history, model_path = train_multiclass_model(
                model=model,
                train_loader=train_loader,
                val_loader=val_loader,
                input_dim=train_dataset.feature_dim,
                model_name=model_name,
                logger=logger
            )

            all_histories[model_name] = history
            all_model_paths[model_name] = model_path

            logger.info(f"加载最佳模型进行测试集评估...")
            checkpoint = torch.load(model_path, map_location=config.DEVICE)
            model.load_state_dict(checkpoint['model_state_dict'])

            eval_res = evaluate_multiclass_model(
                model=model,
                test_loader=test_loader,
                model_name=model_name,
                logger=logger
            )
            all_results[model_name] = eval_res

            plot_multiclass_results(
                history=history,
                eval_results=eval_res,
                model_name=model_name,
                logger=logger
            )

        summary_df, best_model_name = create_model_comparison_table(all_results, logger=logger)
        final_results = {
            'training_time': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'config': {k: v for k, v in config.__dict__.items() if not k.startswith('_')},
            'model_comparison': summary_df.to_dict('records'),
            'best_model': best_model_name,
            'detailed_results': all_results,
            'training_histories': all_histories
        }

        results_path = os.path.join(config.OUTPUT_DIR, "final_results.json")
        with open(results_path, 'w', encoding='utf-8') as f:
            json.dump(final_results, f, ensure_ascii=False, indent=2)
        logger.info(f"\n所有结果已保存: {results_path}")

        logger.info(f"\n{'=' * 100} 训练完成 {'=' * 100}")
        logger.info(f"输出目录: {config.OUTPUT_DIR}")
        logger.info(f"包含内容:")
        logger.info(f"  - models/: 最佳模型文件")
        logger.info(f"  - figures/: 可视化结果图")
        logger.info(f"  - logs/: 训练日志")
        logger.info(f"  - tensorboard/: TensorBoard日志")
        logger.info(f"  - model_comparison.csv: 模型对比表格")
        logger.info(f"  - final_results.json: 完整结果数据")

        return final_results

    except Exception as e:
        logger.error(f"程序执行出错: {str(e)}", exc_info=True)
        raise


def predict_main(
        csv_path: str,
        model_path: str,
        exclude_cols: Optional[List[str]] = None
):
    """预测脚本"""
    logger.info(f"\n{'=' * 100} 糖尿病Raman光谱四分类预测开始 {'=' * 100}")
    logger.info(f"预测时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    try:
        results = predict_new_samples(
            csv_path=csv_path,
            model_path=model_path,
            exclude_cols=exclude_cols,
            logger=logger
        )
        logger.info(f"\n{'=' * 100} 预测完成 {'=' * 100}")
        return results
    except Exception as e:
        logger.error(f"预测出错: {str(e)}", exc_info=True)
        raise


if __name__ == "__main__":
    main()

    # 预测模式
    # predict_main(
    #     csv_path="new_samples.csv",
    #     model_path="diabetes_results_20251118_123456/models/best_MLP_Baseline.pth",
    #     exclude_cols=['Sample', 'Group']
    # )