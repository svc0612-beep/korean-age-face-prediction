from pathlib import Path
import random
import time

import numpy as np
import pandas as pd
from PIL import Image

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from torchvision import models, transforms


PROJECT_DIR = Path(r"C:\Users\svc06\OneDrive\Desktop\한국인_이미지_프로젝트")
DATA_DIR = PROJECT_DIR / "data" / "aging_face"
TRAIN_CSV = DATA_DIR / "train_face_cache_224.csv"
VAL_CSV = DATA_DIR / "val_face_cache_224.csv"
MODEL_PATH = DATA_DIR / "best_age_mobilenet_v2_regression.pt"
HISTORY_PATH = DATA_DIR / "age_mobilenet_v2_history.csv"

SEED = 42
BATCH_SIZE = 32
NUM_WORKERS = 0
MAX_EPOCHS = 100
PATIENCE = 14
MIN_DELTA = 0.02
LR_HEAD = 8e-4
LR_FINETUNE = 8e-5
WEIGHT_DECAY = 3e-4
FREEZE_EPOCHS = 4
LOG_EVERY_BATCHES = 100


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def clean_df(df):
    df = df[["cache_path", "age_past"]].dropna().copy()
    df["age_past"] = pd.to_numeric(df["age_past"], errors="coerce")
    df = df.dropna(subset=["age_past"]).copy()
    df = df[(df["age_past"] >= 0) & (df["age_past"] <= 100)].copy()
    df = df[df["cache_path"].astype(str).map(lambda p: Path(p).exists())].copy()
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
        target = (age - self.target_mean) / self.target_std
        return image, torch.tensor(target, dtype=torch.float32), torch.tensor(age, dtype=torch.float32)


class AgeRegressorV2(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = models.mobilenet_v3_small(weights=models.MobileNet_V3_Small_Weights.DEFAULT)
        in_features = self.backbone.classifier[0].in_features
        self.backbone.classifier = nn.Sequential(
            nn.Linear(in_features, 512),
            nn.Hardswish(inplace=True),
            nn.BatchNorm1d(512),
            nn.Dropout(0.35),
            nn.Linear(512, 128),
            nn.Hardswish(inplace=True),
            nn.Dropout(0.20),
            nn.Linear(128, 1),
        )

    def forward(self, x):
        return self.backbone(x).squeeze(1)


def set_backbone_trainable(model, trainable):
    for param in model.backbone.features.parameters():
        param.requires_grad = trainable


def make_loaders(train_df, val_df, target_mean, target_std):
    train_transform = transforms.Compose(
        [
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=7),
            transforms.RandomAffine(degrees=0, translate=(0.04, 0.04), scale=(0.94, 1.06)),
            transforms.ColorJitter(brightness=0.16, contrast=0.16, saturation=0.10),
            transforms.ToTensor(),
            transforms.RandomErasing(p=0.12, scale=(0.01, 0.04), ratio=(0.5, 2.0)),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    val_transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    train_dataset = AgeCacheDataset(train_df, target_mean, target_std, train_transform)
    val_dataset = AgeCacheDataset(val_df, target_mean, target_std, val_transform)

    age_bins = train_df["age_past"].astype(int).clip(0, 80) // 10
    bin_counts = age_bins.value_counts().sort_index()
    bin_weights = len(train_df) / (len(bin_counts) * bin_counts.clip(lower=1))
    sample_weights = age_bins.map(bin_weights.to_dict()).astype(float).to_numpy()

    sampler = WeightedRandomSampler(
        weights=torch.tensor(sample_weights, dtype=torch.double),
        num_samples=len(sample_weights),
        replacement=True,
    )

    return (
        DataLoader(train_dataset, batch_size=BATCH_SIZE, sampler=sampler, num_workers=NUM_WORKERS),
        DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS),
    )


def make_optimizer(model, lr):
    return torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=lr,
        weight_decay=WEIGHT_DECAY,
    )


def denormalize(pred, mean, std):
    return pred * std + mean


def train_one_epoch(model, loader, criterion, optimizer, device, epoch, target_mean, target_std):
    model.train()
    total_loss = 0.0
    total_abs_error = 0.0
    total = 0

    for batch_idx, (images, targets, ages) in enumerate(loader, start=1):
        images = images.to(device)
        targets = targets.to(device)
        ages = ages.to(device)

        optimizer.zero_grad(set_to_none=True)
        preds_norm = model(images)
        loss = criterion(preds_norm, targets)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=3.0)
        optimizer.step()

        preds_age = denormalize(preds_norm.detach(), target_mean, target_std).clamp(0, 100)
        batch_size = images.size(0)
        total_loss += loss.item() * batch_size
        total_abs_error += torch.abs(preds_age - ages).sum().item()
        total += batch_size

        if batch_idx == 1 or batch_idx % LOG_EVERY_BATCHES == 0 or batch_idx == len(loader):
            print(
                f"epoch={epoch:03d} batch={batch_idx:04d}/{len(loader):04d} "
                f"running_mae={total_abs_error / total:.3f}",
                flush=True,
            )

    return total_loss / total, total_abs_error / total


@torch.no_grad()
def evaluate(model, loader, criterion, device, target_mean, target_std):
    model.eval()
    total_loss = 0.0
    total_abs_error = 0.0
    total = 0

    for images, targets, ages in loader:
        images = images.to(device)
        targets = targets.to(device)
        ages = ages.to(device)
        preds_norm = model(images)
        loss = criterion(preds_norm, targets)
        preds_age = denormalize(preds_norm, target_mean, target_std).clamp(0, 100)

        batch_size = images.size(0)
        total_loss += loss.item() * batch_size
        total_abs_error += torch.abs(preds_age - ages).sum().item()
        total += batch_size

    return total_loss / total, total_abs_error / total


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

    train_loader, val_loader = make_loaders(train_df, val_df, target_mean, target_std)
    model = AgeRegressorV2().to(device)
    set_backbone_trainable(model, False)

    criterion = nn.SmoothL1Loss(beta=0.25)
    optimizer = make_optimizer(model, LR_HEAD)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=5
    )

    best_val_mae = float("inf")
    no_meaningful_improve = 0
    fine_tuned = False
    history = []

    for epoch in range(1, MAX_EPOCHS + 1):
        if epoch == FREEZE_EPOCHS + 1 and not fine_tuned:
            print("unfreeze_backbone=True", flush=True)
            set_backbone_trainable(model, True)
            optimizer = make_optimizer(model, LR_FINETUNE)
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, mode="min", factor=0.5, patience=5
            )
            fine_tuned = True

        start = time.time()
        train_loss, train_mae = train_one_epoch(
            model, train_loader, criterion, optimizer, device, epoch, target_mean, target_std
        )
        val_loss, val_mae = evaluate(model, val_loader, criterion, device, target_mean, target_std)
        scheduler.step(val_mae)
        elapsed = time.time() - start

        improved = val_mae < best_val_mae
        meaningful_improvement = val_mae < (best_val_mae - MIN_DELTA)
        if improved:
            best_val_mae = val_mae
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "target_mean": target_mean,
                    "target_std": target_std,
                    "best_val_mae": best_val_mae,
                    "epoch": epoch,
                    "model_name": "mobilenet_v3_small_v2",
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
            "val_loss": val_loss,
            "val_mae": val_mae,
            "best_val_mae": best_val_mae,
            "lr": optimizer.param_groups[0]["lr"],
            "elapsed_sec": elapsed,
            "improved": improved,
            "meaningful_improvement": meaningful_improvement,
            "fine_tuned": fine_tuned,
        }
        history.append(row)
        pd.DataFrame(history).to_csv(HISTORY_PATH, index=False, encoding="utf-8-sig")

        print(
            f"epoch={epoch:03d}/{MAX_EPOCHS} train_mae={train_mae:.3f} "
            f"val_mae={val_mae:.3f} best={best_val_mae:.3f} "
            f"lr={optimizer.param_groups[0]['lr']:.2e} improved={improved} "
            f"no_improve={no_meaningful_improve}/{PATIENCE} time={elapsed:.1f}s",
            flush=True,
        )

        if no_meaningful_improve >= PATIENCE:
            print("early_stopping=True", flush=True)
            break

    print(f"done best_val_mae={best_val_mae:.3f}", flush=True)
    print(f"saved_model={MODEL_PATH}", flush=True)
    print(f"saved_history={HISTORY_PATH}", flush=True)


if __name__ == "__main__":
    main()
