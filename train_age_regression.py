from pathlib import Path
import math
import random
import time

import numpy as np
import pandas as pd
from PIL import Image

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from torchvision import transforms


PROJECT_DIR = Path(r"C:\Users\svc06\OneDrive\Desktop\한국인_이미지_프로젝트")
DATA_DIR = PROJECT_DIR / "data" / "aging_face"
TRAIN_CSV = DATA_DIR / "train_dataset.csv"
VAL_CSV = DATA_DIR / "val_dataset.csv"
MODEL_PATH = DATA_DIR / "best_age_regression_model.pt"
HISTORY_PATH = DATA_DIR / "age_regression_history.csv"

SEED = 42
IMG_SIZE = 128
BATCH_SIZE = 64
NUM_WORKERS = 0
MAX_EPOCHS = 100
PATIENCE = 10
MIN_DELTA = 0.05
LR = 1e-3
WEIGHT_DECAY = 1e-4
TRAIN_LIMIT = 1000
VAL_LIMIT = 200
LOG_EVERY_BATCHES = 5


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


class AgingFaceRegressionDataset(Dataset):
    def __init__(self, df: pd.DataFrame, transform=None):
        self.df = df.reset_index(drop=True).copy()
        self.transform = transform

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]
        image = Image.open(row["image_path"]).convert("RGB")
        if self.transform is not None:
            image = self.transform(image)
        age = torch.tensor(float(row["age_past"]), dtype=torch.float32)
        return image, age


class AgeCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.regressor = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.3),
            nn.Linear(256, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 1),
        )

    def forward(self, x):
        return self.regressor(self.features(x)).squeeze(1)


def make_age_bins(ages: pd.Series) -> pd.Series:
    bins = [0, 10, 20, 30, 40, 50, 60, math.inf]
    labels = list(range(len(bins) - 1))
    return pd.cut(ages, bins=bins, labels=labels, right=False, include_lowest=True).astype(int)


def build_loaders(train_df: pd.DataFrame, val_df: pd.DataFrame):
    train_transform = transforms.Compose(
        [
            transforms.Resize((IMG_SIZE, IMG_SIZE)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.08),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    val_transform = transforms.Compose(
        [
            transforms.Resize((IMG_SIZE, IMG_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    train_dataset = AgingFaceRegressionDataset(train_df, train_transform)
    val_dataset = AgingFaceRegressionDataset(val_df, val_transform)

    age_bins = make_age_bins(train_df["age_past"])
    bin_counts = age_bins.value_counts().sort_index()
    bin_weights = len(train_df) / (len(bin_counts) * bin_counts.clip(lower=1))
    sample_weights = age_bins.map(bin_weights.to_dict()).astype(float).to_numpy()

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
    total_loss = 0.0
    total_abs_error = 0.0
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
    total_loss = 0.0
    total_abs_error = 0.0
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


def main():
    seed_everything(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device={device}", flush=True)
    print(f"train_csv={TRAIN_CSV}", flush=True)
    print(f"val_csv={VAL_CSV}", flush=True)

    train_df = pd.read_csv(TRAIN_CSV)
    val_df = pd.read_csv(VAL_CSV)
    train_df = train_df[["image_path", "age_past"]].dropna().copy()
    val_df = val_df[["image_path", "age_past"]].dropna().copy()
    train_df["age_past"] = train_df["age_past"].astype(float)
    val_df["age_past"] = val_df["age_past"].astype(float)

    if TRAIN_LIMIT:
        train_df = train_df.sample(n=min(TRAIN_LIMIT, len(train_df)), random_state=SEED).reset_index(drop=True)
    if VAL_LIMIT:
        val_df = val_df.sample(n=min(VAL_LIMIT, len(val_df)), random_state=SEED).reset_index(drop=True)

    print(f"train_rows={len(train_df)} val_rows={len(val_df)}", flush=True)
    print("train age summary")
    print(train_df["age_past"].describe(), flush=True)
    print("val age summary")
    print(val_df["age_past"].describe(), flush=True)

    train_loader, val_loader = build_loaders(train_df, val_df)

    images, ages = next(iter(train_loader))
    print(f"sanity_batch images={tuple(images.shape)} ages={tuple(ages.shape)}", flush=True)

    model = AgeCNN().to(device)
    criterion = nn.L1Loss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=3
    )

    best_val_mae = float("inf")
    epochs_without_improvement = 0
    history = []

    for epoch in range(1, MAX_EPOCHS + 1):
        start = time.time()
        train_loss, train_mae = train_one_epoch(model, train_loader, criterion, optimizer, device, epoch)
        val_loss, val_mae = evaluate(model, val_loader, criterion, device)
        scheduler.step(val_mae)
        elapsed = time.time() - start

        improved = val_mae < (best_val_mae - MIN_DELTA)
        if improved:
            best_val_mae = val_mae
            epochs_without_improvement = 0
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "img_size": IMG_SIZE,
                    "best_val_mae": best_val_mae,
                    "epoch": epoch,
                    "mean": [0.485, 0.456, 0.406],
                    "std": [0.229, 0.224, 0.225],
                },
                MODEL_PATH,
            )
        else:
            epochs_without_improvement += 1

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
        }
        history.append(row)
        pd.DataFrame(history).to_csv(HISTORY_PATH, index=False, encoding="utf-8-sig")

        print(
            f"epoch={epoch:03d}/{MAX_EPOCHS} "
            f"train_mae={train_mae:.3f} val_mae={val_mae:.3f} "
            f"best={best_val_mae:.3f} lr={optimizer.param_groups[0]['lr']:.2e} "
            f"improved={improved} no_improve={epochs_without_improvement}/{PATIENCE} "
            f"time={elapsed:.1f}s",
            flush=True,
        )

        if epochs_without_improvement >= PATIENCE:
            print("early_stopping=True", flush=True)
            break

    print(f"done best_val_mae={best_val_mae:.3f}", flush=True)
    print(f"saved_model={MODEL_PATH}", flush=True)
    print(f"saved_history={HISTORY_PATH}", flush=True)


if __name__ == "__main__":
    main()
