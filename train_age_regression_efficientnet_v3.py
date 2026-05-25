from pathlib import Path
import random
import time

import numpy as np
import pandas as pd
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from torchvision import models, transforms


PROJECT_DIR = Path(__file__).resolve().parent
DATA_DIR = PROJECT_DIR / "data" / "aging_face"
TRAIN_CSV = DATA_DIR / "train_face_cache_224.csv"
VAL_CSV = DATA_DIR / "val_face_cache_224.csv"
MODEL_PATH = DATA_DIR / "best_age_efficientnet_v3_multitask.pt"
HISTORY_PATH = DATA_DIR / "age_efficientnet_v3_history.csv"
VAL_PRED_PATH = DATA_DIR / "age_efficientnet_v3_val_predictions.csv"

SEED = 42
BATCH_SIZE = 24
NUM_WORKERS = 0
MAX_EPOCHS = 60
PATIENCE = 10
MIN_DELTA = 0.02
FREEZE_EPOCHS = 3
LR_HEAD = 7e-4
LR_FINETUNE = 6e-5
WEIGHT_DECAY = 4e-4
CLASS_LOSS_WEIGHT = 0.35
ADULT_LOSS_WEIGHT = 1.35
OLDER_LOSS_WEIGHT = 1.75
LOG_EVERY_BATCHES = 100

AGE_GROUPS = ["0s", "10s", "20s", "30s", "40s", "50s", "60plus"]


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def age_to_group_idx(age):
    age = int(age)
    if age >= 60:
        return 6
    return max(0, min(5, age // 10))


def clean_df(df):
    df = df[["cache_path", "age_past", "gender"]].copy()
    df["age_past"] = pd.to_numeric(df["age_past"], errors="coerce")
    df = df.dropna(subset=["cache_path", "age_past"]).copy()
    df = df[(df["age_past"] >= 0) & (df["age_past"] <= 100)].copy()
    df = df[df["cache_path"].astype(str).map(lambda p: Path(p).exists())].copy()
    df["age_group_idx"] = df["age_past"].map(age_to_group_idx).astype(int)
    return df.reset_index(drop=True)


class AgeCacheDataset(Dataset):
    def __init__(self, df, target_mean, target_std, transform=None):
        self.df = df.reset_index(drop=True).copy()
        self.target_mean = float(target_mean)
        self.target_std = float(target_std)
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        image = Image.open(row["cache_path"]).convert("RGB")
        if self.transform:
            image = self.transform(image)

        age = float(row["age_past"])
        target_norm = (age - self.target_mean) / self.target_std
        group = int(row["age_group_idx"])
        sample_weight = 1.0
        if age >= 40:
            sample_weight = OLDER_LOSS_WEIGHT
        elif age >= 30:
            sample_weight = ADULT_LOSS_WEIGHT

        return (
            image,
            torch.tensor(target_norm, dtype=torch.float32),
            torch.tensor(age, dtype=torch.float32),
            torch.tensor(group, dtype=torch.long),
            torch.tensor(sample_weight, dtype=torch.float32),
        )


class AgeEfficientNetMultiTask(nn.Module):
    def __init__(self, num_groups=len(AGE_GROUPS)):
        super().__init__()
        try:
            weights = models.EfficientNet_B0_Weights.DEFAULT
            self.backbone = models.efficientnet_b0(weights=weights)
            print("efficientnet_weights=DEFAULT", flush=True)
        except Exception as exc:
            print(f"efficientnet_weights=None reason={exc}", flush=True)
            self.backbone = models.efficientnet_b0(weights=None)

        in_features = self.backbone.classifier[1].in_features
        self.backbone.classifier = nn.Identity()
        self.shared = nn.Sequential(
            nn.Linear(in_features, 512),
            nn.SiLU(inplace=True),
            nn.BatchNorm1d(512),
            nn.Dropout(0.35),
            nn.Linear(512, 256),
            nn.SiLU(inplace=True),
            nn.Dropout(0.25),
        )
        self.age_head = nn.Linear(256, 1)
        self.group_head = nn.Linear(256, num_groups)

    def forward(self, x):
        features = self.backbone(x)
        hidden = self.shared(features)
        age_norm = self.age_head(hidden).squeeze(1)
        group_logits = self.group_head(hidden)
        return age_norm, group_logits


def set_backbone_trainable(model, trainable):
    for param in model.backbone.features.parameters():
        param.requires_grad = trainable


def make_transforms():
    train_transform = transforms.Compose(
        [
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=8),
            transforms.RandomAffine(degrees=0, translate=(0.05, 0.05), scale=(0.90, 1.12)),
            transforms.ColorJitter(brightness=0.22, contrast=0.24, saturation=0.12, hue=0.02),
            transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 1.2)),
            transforms.ToTensor(),
            transforms.RandomErasing(p=0.18, scale=(0.01, 0.05), ratio=(0.4, 2.5)),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    val_transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    return train_transform, val_transform


def make_loaders(train_df, val_df, target_mean, target_std):
    train_transform, val_transform = make_transforms()
    train_dataset = AgeCacheDataset(train_df, target_mean, target_std, train_transform)
    val_dataset = AgeCacheDataset(val_df, target_mean, target_std, val_transform)

    group_counts = train_df["age_group_idx"].value_counts().sort_index()
    group_weights = len(train_df) / (len(group_counts) * group_counts.clip(lower=1))
    sample_weights = train_df["age_group_idx"].map(group_weights.to_dict()).astype(float).to_numpy().copy()
    sample_weights *= np.where(train_df["age_past"].to_numpy() >= 40, 1.45, 1.0)
    sample_weights *= np.where((train_df["age_past"].to_numpy() >= 30) & (train_df["age_past"].to_numpy() < 40), 1.20, 1.0)

    sampler = WeightedRandomSampler(
        weights=torch.tensor(sample_weights, dtype=torch.double),
        num_samples=len(sample_weights),
        replacement=True,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        sampler=sampler,
        num_workers=NUM_WORKERS,
        pin_memory=torch.cuda.is_available(),
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=torch.cuda.is_available(),
    )
    return train_loader, val_loader


def make_optimizer(model, lr):
    return torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=lr,
        weight_decay=WEIGHT_DECAY,
    )


def denormalize(pred_norm, mean, std):
    return pred_norm * std + mean


def multitask_loss(pred_norm, group_logits, targets_norm, groups, sample_weights):
    reg_loss = F.smooth_l1_loss(pred_norm, targets_norm, beta=0.25, reduction="none")
    reg_loss = (reg_loss * sample_weights).mean()
    cls_loss = F.cross_entropy(group_logits, groups)
    return reg_loss + CLASS_LOSS_WEIGHT * cls_loss, reg_loss.detach(), cls_loss.detach()


def train_one_epoch(model, loader, optimizer, device, epoch, target_mean, target_std):
    model.train()
    total_loss = 0.0
    total_abs_error = 0.0
    total_group_correct = 0
    total = 0

    for batch_idx, (images, targets_norm, ages, groups, sample_weights) in enumerate(loader, start=1):
        images = images.to(device)
        targets_norm = targets_norm.to(device)
        ages = ages.to(device)
        groups = groups.to(device)
        sample_weights = sample_weights.to(device)

        optimizer.zero_grad(set_to_none=True)
        pred_norm, group_logits = model(images)
        loss, _, _ = multitask_loss(pred_norm, group_logits, targets_norm, groups, sample_weights)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=3.0)
        optimizer.step()

        pred_age = denormalize(pred_norm.detach(), target_mean, target_std).clamp(0, 100)
        batch_size = images.size(0)
        total_loss += loss.item() * batch_size
        total_abs_error += torch.abs(pred_age - ages).sum().item()
        total_group_correct += (group_logits.detach().argmax(dim=1) == groups).sum().item()
        total += batch_size

        if batch_idx == 1 or batch_idx % LOG_EVERY_BATCHES == 0 or batch_idx == len(loader):
            print(
                f"epoch={epoch:03d} batch={batch_idx:04d}/{len(loader):04d} "
                f"running_mae={total_abs_error / total:.3f} "
                f"running_group_acc={total_group_correct / total:.3f}",
                flush=True,
            )

    return total_loss / total, total_abs_error / total, total_group_correct / total


@torch.no_grad()
def evaluate(model, loader, device, target_mean, target_std):
    model.eval()
    total_loss = 0.0
    total_abs_error = 0.0
    total_group_correct = 0
    total = 0
    rows = []

    for images, targets_norm, ages, groups, sample_weights in loader:
        images = images.to(device)
        targets_norm = targets_norm.to(device)
        ages = ages.to(device)
        groups = groups.to(device)
        sample_weights = sample_weights.to(device)

        pred_norm, group_logits = model(images)
        loss, _, _ = multitask_loss(pred_norm, group_logits, targets_norm, groups, sample_weights)
        pred_age = denormalize(pred_norm, target_mean, target_std).clamp(0, 100)
        pred_group = group_logits.argmax(dim=1)

        batch_size = images.size(0)
        total_loss += loss.item() * batch_size
        total_abs_error += torch.abs(pred_age - ages).sum().item()
        total_group_correct += (pred_group == groups).sum().item()
        total += batch_size

        for true_age, pred, true_group, pred_group_idx in zip(
            ages.cpu().numpy(),
            pred_age.cpu().numpy(),
            groups.cpu().numpy(),
            pred_group.cpu().numpy(),
        ):
            rows.append(
                {
                    "true_age": float(true_age),
                    "pred_age": float(pred),
                    "true_group": AGE_GROUPS[int(true_group)],
                    "pred_group": AGE_GROUPS[int(pred_group_idx)],
                    "abs_error": abs(float(pred) - float(true_age)),
                    "bias": float(pred) - float(true_age),
                }
            )

    pred_df = pd.DataFrame(rows)
    adult_mae = pred_df[pred_df["true_age"] >= 30]["abs_error"].mean()
    return (
        total_loss / total,
        total_abs_error / total,
        total_group_correct / total,
        float(adult_mae),
        pred_df,
    )


def summarize_by_group(pred_df):
    summary = (
        pred_df.groupby("true_group")
        .agg(
            n=("true_age", "size"),
            true_mean=("true_age", "mean"),
            pred_mean=("pred_age", "mean"),
            mae=("abs_error", "mean"),
            bias=("bias", "mean"),
        )
        .reset_index()
    )
    return summary


def main():
    seed_everything(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device={device}", flush=True)

    train_df = clean_df(pd.read_csv(TRAIN_CSV))
    val_df = clean_df(pd.read_csv(VAL_CSV))
    target_mean = float(train_df["age_past"].mean())
    target_std = float(train_df["age_past"].std())
    print(f"train_rows={len(train_df)} val_rows={len(val_df)}", flush=True)
    print(f"target_mean={target_mean:.4f} target_std={target_std:.4f}", flush=True)
    print("train_group_counts=", train_df["age_group_idx"].value_counts().sort_index().to_dict(), flush=True)

    train_loader, val_loader = make_loaders(train_df, val_df, target_mean, target_std)
    model = AgeEfficientNetMultiTask().to(device)
    set_backbone_trainable(model, False)

    optimizer = make_optimizer(model, LR_HEAD)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=4
    )

    best_score = float("inf")
    best_val_mae = float("inf")
    best_adult_mae = float("inf")
    no_meaningful_improve = 0
    fine_tuned = False
    history = []

    for epoch in range(1, MAX_EPOCHS + 1):
        if epoch == FREEZE_EPOCHS + 1 and not fine_tuned:
            print("unfreeze_backbone=True", flush=True)
            set_backbone_trainable(model, True)
            optimizer = make_optimizer(model, LR_FINETUNE)
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, mode="min", factor=0.5, patience=4
            )
            fine_tuned = True

        start = time.time()
        train_loss, train_mae, train_group_acc = train_one_epoch(
            model, train_loader, optimizer, device, epoch, target_mean, target_std
        )
        val_loss, val_mae, val_group_acc, adult_mae, pred_df = evaluate(
            model, val_loader, device, target_mean, target_std
        )

        score = 0.65 * val_mae + 0.35 * adult_mae
        scheduler.step(score)
        elapsed = time.time() - start

        improved = score < best_score
        meaningful_improvement = score < (best_score - MIN_DELTA)
        if improved:
            best_score = score
            best_val_mae = val_mae
            best_adult_mae = adult_mae
            pred_df.to_csv(VAL_PRED_PATH, index=False, encoding="utf-8-sig")
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "target_mean": target_mean,
                    "target_std": target_std,
                    "best_score": best_score,
                    "best_val_mae": best_val_mae,
                    "best_adult_mae": best_adult_mae,
                    "epoch": epoch,
                    "model_name": "efficientnet_b0_v3_multitask",
                    "age_groups": AGE_GROUPS,
                    "mean": [0.485, 0.456, 0.406],
                    "std": [0.229, 0.224, 0.225],
                },
                MODEL_PATH,
            )

        if meaningful_improvement:
            no_meaningful_improve = 0
        else:
            no_meaningful_improve += 1

        row = {
            "epoch": epoch,
            "train_loss": train_loss,
            "train_mae": train_mae,
            "train_group_acc": train_group_acc,
            "val_loss": val_loss,
            "val_mae": val_mae,
            "val_group_acc": val_group_acc,
            "adult_mae": adult_mae,
            "score": score,
            "best_score": best_score,
            "best_val_mae": best_val_mae,
            "best_adult_mae": best_adult_mae,
            "lr": optimizer.param_groups[0]["lr"],
            "elapsed_sec": elapsed,
            "improved": improved,
            "meaningful_improvement": meaningful_improvement,
            "fine_tuned": fine_tuned,
        }
        history.append(row)
        pd.DataFrame(history).to_csv(HISTORY_PATH, index=False, encoding="utf-8-sig")

        print(
            f"epoch={epoch:03d}/{MAX_EPOCHS} "
            f"train_mae={train_mae:.3f} val_mae={val_mae:.3f} "
            f"adult_mae={adult_mae:.3f} group_acc={val_group_acc:.3f} "
            f"score={score:.3f} best={best_score:.3f} "
            f"lr={optimizer.param_groups[0]['lr']:.2e} improved={improved} "
            f"no_improve={no_meaningful_improve}/{PATIENCE} time={elapsed:.1f}s",
            flush=True,
        )

        if improved:
            print("group_summary=", flush=True)
            print(summarize_by_group(pred_df).to_string(index=False), flush=True)

        if no_meaningful_improve >= PATIENCE:
            print("early_stopping=True", flush=True)
            break

    print(f"done best_score={best_score:.3f}", flush=True)
    print(f"best_val_mae={best_val_mae:.3f} best_adult_mae={best_adult_mae:.3f}", flush=True)
    print(f"saved_model={MODEL_PATH}", flush=True)
    print(f"saved_history={HISTORY_PATH}", flush=True)
    print(f"saved_val_predictions={VAL_PRED_PATH}", flush=True)


if __name__ == "__main__":
    main()
