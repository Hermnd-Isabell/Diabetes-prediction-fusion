import argparse
import yaml
import torch
from torch.utils.data import DataLoader
import pandas as pd

# 导入我们自己写的模块
from datasets.raman_dataset import RamanDataset, collate_fn, preprocess_spectrum
from trainers.trainer import Trainer

# 各类模型
from models.Baseline import SpectraEncoder, TabularEncoder, ConcatFusion, EnsembleFusion
from models.attention_models import AttentionMultimodal
from models.tft_models import TFTMultimodal
from models.mmtm_models import MMTMMultimodal

# ----------------------------
# 1️⃣ 读取配置
# ----------------------------
def load_config(config_path):
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


# ----------------------------
# 2️⃣ 模型构建函数
# ----------------------------
def build_model(cfg, tab_dim, spec_len):
    """根据 YAML 中的 model.name 构造模型"""
    model_name = cfg["model"]["name"]

    if model_name == "Spectra-only":
        return SpectraEncoder(input_dim=spec_len, hidden_dim=256)
    elif model_name == "Clinical-only":
        return TabularEncoder(input_dim=tab_dim, hidden_dim=128)
    elif model_name == "ConcatFusion":
        return ConcatFusion(spec_dim=256, clin_dim=128)
    elif model_name == "EnsembleFusion":
        return EnsembleFusion(spec_dim=256, clin_dim=128)
    elif model_name == "MultimodalAttention":
        spec_cfg = {"base_filters": 32, "emb_dim": 256, "use_transformer": False, "tab_emb": 128}
        return MultimodalModel(
            spec_cfg=spec_cfg,
            tab_dim=tab_dim,
            num_classes=cfg["model"]["num_classes"],
            fusion=cfg["model"].get("fusion", "cross")
        )
    elif model_name == "TFTMultimodal":
        return TFTMultimodal(
            tab_dim=tab_dim,
            spec_len=spec_len,
            spec_emb=cfg["model"].get("spec_emb", 256),
            tab_emb=cfg["model"].get("tab_emb", 128),
            num_classes=cfg["model"]["num_classes"]
        )
    elif model_name == "MMTMMultimodal":
        return MMTMMultimodal(
            tab_dim=tab_dim,
            spec_len=spec_len,
            spec_emb=cfg["model"].get("spec_emb", 256),
            tab_emb=cfg["model"].get("tab_emb", 128),
            num_classes=cfg["model"]["num_classes"],
            mmtm_bottleneck=cfg["model"].get("mmtm_bottleneck", 128)
        )
    else:
        raise ValueError(f"❌ Unknown model name: {model_name}")


# ----------------------------
# 3️⃣ 主函数：读取数据 + 训练
# ----------------------------
def main(args):
    # 加载配置
    cfg = load_config(args.config)

    # ---------------- 数据准备 ----------------
    spectra_csv = cfg["data"]["spectra_csv"]
    clinical_csv = cfg["data"]["clinical_csv"]

    spectra_df = pd.read_csv(spectra_csv, sep=None, engine="python")
    wave_cols = [c for c in spectra_df.columns if c not in ["Sample", "Group"]]

    dataset = RamanDataset(
        spectra_csv,
        clinical_csv,
        wave_cols,
        label_col=cfg["data"].get("label_col", "Group"),
        preprocess_fn=preprocess_spectrum,
        min_scans=1,
        max_scans=cfg["data"].get("max_scans", 180)
    )

    # ---------------- 数据划分 ----------------
    from torch.utils.data import random_split
    train_size = int(0.7 * len(dataset))
    val_size = len(dataset) - train_size
    train_set, val_set = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(
        train_set,
        batch_size=cfg["train"]["batch_size"],
        shuffle=True,
        collate_fn=collate_fn
    )
    val_loader = DataLoader(
        val_set,
        batch_size=cfg["train"]["batch_size"],
        shuffle=False,
        collate_fn=collate_fn
    )

    # ---------------- 模型构建 ----------------
    tab_dim = dataset.items[0]["tabular"].shape[0]
    spec_len = len(wave_cols)
    model = build_model(cfg, tab_dim=tab_dim, spec_len=spec_len)

    # ---------------- 训练器 ----------------
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    trainer = Trainer(
        model,
        device=device,
        lr=cfg["train"]["lr"],
        weight_decay=cfg["train"].get("weight_decay", 1e-4)
    )

    # ---------------- 训练循环 ----------------
    for epoch in range(cfg["train"]["epochs"]):
        train_loss = trainer.train_epoch(train_loader)
        val_loss, metrics = trainer.eval_epoch(val_loader)
        print(f"Epoch {epoch}: train_loss={train_loss:.4f}, val_loss={val_loss:.4f}, metrics={metrics}")

        # 保存 checkpoint
        save_prefix = cfg["train"].get("save_prefix", "checkpoint")
        torch.save({
            "epoch": epoch,
            "model_state": model.state_dict(),
            "opt_state": trainer.optimizer.state_dict(),
            "metrics": metrics
        }, f"{save_prefix}_epoch{epoch}.pt")


# ----------------------------
# 4️⃣ 启动命令
# ----------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True, help="Path to yaml config file")
    args = parser.parse_args()
    main(args)
