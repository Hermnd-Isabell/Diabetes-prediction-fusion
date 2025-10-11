import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import numpy as np
from sklearn.metrics import roc_auc_score, f1_score, roc_curve, accuracy_score


# ----------------------------
# Metrics
# ----------------------------
def evaluate_metrics(y_true, y_prob, threshold=0.5):
    y_true = np.array(y_true)
    y_prob = np.array(y_prob)

    auc = roc_auc_score(y_true, y_prob)
    y_pred = (y_prob >= threshold).astype(int)
    f1 = f1_score(y_true, y_pred)
    acc = accuracy_score(y_true, y_pred)

    fpr, tpr, thresholds = roc_curve(y_true, y_prob)
    specificity = 1 - fpr
    mask = specificity >= 0.9
    sens_at_90 = tpr[mask].max() if np.any(mask) else np.nan

    return {"AUC": auc, "F1": f1, "ACC": acc, "sensitivity@90%spec": sens_at_90}


# ----------------------------
# Trainer Class
# ----------------------------
class Trainer:
    def __init__(self, model, device="cpu", lr=1e-3, weight_decay=1e-4):
        self.model = model.to(device)
        self.device = device
        self.criterion = nn.CrossEntropyLoss()  # for multi-class (num_classes >=2)
        self.optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    def train_epoch(self, loader):
        self.model.train()
        total_loss = 0.0
        for batch in tqdm(loader, desc="train"):
            spectra = batch["spectra"].to(self.device)
            mask = batch.get("mask", None)
            if mask is not None:
                mask = mask.to(self.device)
            tabular = batch["tabular"].to(self.device)
            labels = batch["label"].to(self.device)

            self.optimizer.zero_grad()
            outputs = self.model(spectra, mask, tabular)   # ✅ 模型必须返回 dict
            logits = outputs["logits"]
            loss = self.criterion(logits, labels)
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item() * labels.size(0)
        return total_loss / len(loader.dataset)

    def eval_epoch(self, loader):
        self.model.eval()
        all_true, all_prob = [], []
        total_loss = 0.0
        with torch.no_grad():
            for batch in tqdm(loader, desc="eval"):
                spectra = batch["spectra"].to(self.device)
                mask = batch.get("mask", None)
                if mask is not None:
                    mask = mask.to(self.device)
                tabular = batch["tabular"].to(self.device)
                labels = batch["label"].to(self.device)

                outputs = self.model(spectra, mask, tabular)   # ✅ 模型必须返回 dict
                logits = outputs["logits"]
                loss = self.criterion(logits, labels)
                total_loss += loss.item() * labels.size(0)

                probs = torch.softmax(logits, dim=1)[:, 1]  # class=1 prob
                all_true.extend(labels.cpu().numpy())
                all_prob.extend(probs.cpu().numpy())

        metrics = evaluate_metrics(all_true, all_prob)
        return total_loss / len(loader.dataset), metrics
