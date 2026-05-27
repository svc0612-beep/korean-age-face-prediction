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
MODEL_PATH = DATA_DIR / "best_age_mobilenet_regression.pt"
HISTORY_PATH = DATA_DIR / "age_mobilenet_history.csv"

SEED = 42
IMG_SIZE = 224
BATCH_SIZE = 32
NUM_WORKERS = 0
MAX_EPOCHS = 100
PATIENCE = 12
MIN_DELTA = 0.03
LR_HEAD = 1e-3
LR_FINETUNE = 1e-4
WEIGHT_DECAY = 1e-4
FREEZE_EPOCHS = 5
LOG_EVERY_BATCHES = 100
USE_PRETRAINED = True


def seed_everything(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


class AgeCacheDataset(Dataset):
    def __init__(self, df: pd.DataFrame, transform=None):
        self.df = df.reset_index(drop=True).copy()
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        image = Image.open(row["cache_path"]).convert("RGB")
        if self.transform:
            image = self.transform(image)
        age = torch.tensor(float(row["age_past"]), dtype=torch.float32)
        return image, age


class AgeRegressor(nn.Module):
    def __init__(self, use_pretrained=True):
        super().__init__()
        weights = models.MobileNet_V3_Small_Weights.DEFAULT if use_pretrained else None
        self.backbone = models.mobilenet_v3_small(weights=weights)
        in_features = self.backbone.classifier[0].in_features
        self.backbone.classifier = nn.Sequential(
            nn.Linear(in_features, 256),
            nn.Hardswish(inplace=True),
            nn.Dropout(p=0.2),
            nn.Linear(256, 1),
        )

    def forward(self, x):
        return self.backbone(x).squeeze(1)


def set_backbone_trainable(model: AgeRegressor, trainable: bool):
    for param in model.backbone.features.parameters():
        param.requires_grad = trainable


def make_loaders(train_df, val_df):
    train_transform = transforms.Compose(
        [
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=5),
            transforms.ColorJitter(brightness=0.12, contrast=0.12, saturation=0.08),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    val_transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    train_dataset = AgeCacheDataset(train_df, train_transform)
    val_dataset = AgeCacheDataset(val_df, val_transform)

    age_group = train_df["age_past"].astype(int).clip(0, 80) // 10
    group_counts = age_group.value_counts().sort_index()
    group_weights = len(train_df) / (len(group_counts) * group_counts.clip(lower=1))
    sample_weights = age_group.map(group_weights.to_dict()).astype(float).to_numpy()

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
        pin_memory=False,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=False,
    )
    return train_loader, val_loader


def train_one_epoch(model, loader, criterion, optimizer, device, epoch):
    model.train()
    total_abs_error = 0.0
    total_loss = 0.0
    total = 0

    for batch_idx, (images, ages) in enumerate(loader, start=1):
        images = images.to(device)
        ages = ages.to(device)

        optimizer.zero_grad(set_to_none=True)
        preds = model(images)
        loss = criterion(preds, ages)
        loss.backward()
        optimizer.step()

        batch_size = images.size(0)
        total_loss += loss.item() * batch_size
        total_abs_error += torch.abs(preds.detach() - ages).sum().item()
        total += batch_size

        if batch_idx == 1 or batch_idx % LOG_EVERY_BATCHES == 0 or batch_idx == len(loader):
            print(
                f"epoch={epoch:03d} batch={batch_idx:04d}/{len(loader):04d} "
                f"running_mae={total_abs_error / total:.3f}",
                flush=True,
            )

    return total_loss / total, total_abs_error / total


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    total_abs_error = 0.0
    total_loss = 0.0
    total = 0

    for images, ages in loader:
        images = images.to(device)
        ages = ages.to(device)
        preds = model(images)
        loss = criterion(preds, ages)

        batch_size = images.size(0)
        total_loss += loss.item() * batch_size
        total_abs_error += torch.abs(preds - ages).sum().item()
        total += batch_size

    return total_loss / total, total_abs_error / total


def make_optimizer(model, lr):
    return torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=lr,
        weight_decay=WEIGHT_DECAY,
    )


def main():
    seed_everything(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device={device}", flush=True)
    print(f"train_csv={TRAIN_CSV}", flush=True)
    print(f"val_csv={VAL_CSV}", flush=True)

    train_df = pd.read_csv(TRAIN_CSV)
    val_df = pd.read_csv(VAL_CSV)
    train_df = train_df[["cache_path", "age_past"]].dropna().copy()
    val_df = val_df[["cache_path", "age_past"]].dropna().copy()
    train_df["age_past"] = train_df["age_past"].astype(float)
    val_df["age_past"] = val_df["age_past"].astype(float)

    print(f"train_rows={len(train_df)} val_rows={len(val_df)}", flush=True)
    train_loader, val_loader = make_loaders(train_df, val_df)

    images, ages = next(iter(train_loader))
    print(f"sanity_batch images={tuple(images.shape)} ages={tuple(ages.shape)}", flush=True)

    model = AgeRegressor(use_pretrained=USE_PRETRAINED).to(device)
    set_backbone_trainable(model, False)

    criterion = nn.SmoothL1Loss(beta=3.0)
    optimizer = make_optimizer(model, LR_HEAD)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=4
    )

    best_val_mae = float("inf")
    no_improve = 0
    history = []
    fine_tuned = False

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
        train_loss, train_mae = train_one_epoch(model, train_loader, criterion, optimizer, device, epoch)
        val_loss, val_mae = evaluate(model, val_loader, criterion, device)
        scheduler.step(val_mae)
        elapsed = time.time() - start

        improved = val_mae < best_val_mae
        meaningful_improvement = val_mae < (best_val_mae - MIN_DELTA)
        if improved:
            best_val_mae = val_mae
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "img_size": IMG_SIZE,
                    "best_val_mae": best_val_mae,
                    "epoch": epoch,
                    "model_name": "mobilenet_v3_small",
                    "mean": [0.485, 0.456, 0.406],
                    "std": [0.229, 0.224, 0.225],
                },
                MODEL_PATH,
            )
        if meaningful_improvement:
            no_improve = 0
        else:
            no_improve += 1

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
            f"no_improve={no_improve}/{PATIENCE} time={elapsed:.1f}s",
            flush=True,
        )

        if no_improve >= PATIENCE:
            print("early_stopping=True", flush=True)
            break

    print(f"done best_val_mae={best_val_mae:.3f}", flush=True)
    print(f"saved_model={MODEL_PATH}", flush=True)
    print(f"saved_history={HISTORY_PATH}", flush=True)


if __name__ == "__main__":
    main()
