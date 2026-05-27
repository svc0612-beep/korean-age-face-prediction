from pathlib import Path
import json

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms


PROJECT_DIR = Path(r"C:\Users\svc06\OneDrive\Desktop\한국인_이미지_프로젝트")
DATA_DIR = PROJECT_DIR / "data" / "aging_face"
REPORT_DIR = PROJECT_DIR / "reports"
VAL_CSV = DATA_DIR / "val_face_cache_224.csv"
MODEL_PATH = DATA_DIR / "best_age_mobilenet_regression.pt"
CALIBRATION_PATH = DATA_DIR / "age_calibration.json"
GROUP_REPORT_PATH = REPORT_DIR / "age_error_by_group.csv"
PREDICTION_REPORT_PATH = REPORT_DIR / "val_predictions.csv"


class AgeRegressor(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = models.mobilenet_v3_small(weights=None)
        in_features = self.backbone.classifier[0].in_features
        self.backbone.classifier = nn.Sequential(
            nn.Linear(in_features, 256),
            nn.Hardswish(inplace=True),
            nn.Dropout(p=0.2),
            nn.Linear(256, 1),
        )

    def forward(self, x):
        return self.backbone(x).squeeze(1)


class CacheDataset(Dataset):
    def __init__(self, df, transform):
        self.df = df.reset_index(drop=True)
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        row = self.df.iloc[index]
        image = Image.open(row["cache_path"]).convert("RGB")
        return self.transform(image), float(row["age_past"]), row["cache_path"]


def age_group(age):
    age = float(age)
    if age < 10:
        return "0s"
    if age < 20:
        return "10s"
    if age < 30:
        return "20s"
    if age < 40:
        return "30s"
    if age < 50:
        return "40s"
    if age < 60:
        return "50s"
    return "60plus"


def pred_bin(age):
    age = max(0, min(100, float(age)))
    bucket = int(age // 10) * 10
    return "60plus" if bucket >= 60 else f"{bucket}s"


def main():
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(MODEL_PATH, map_location=device)
    img_size = int(checkpoint.get("img_size", 224))
    mean = checkpoint.get("mean", [0.485, 0.456, 0.406])
    std = checkpoint.get("std", [0.229, 0.224, 0.225])

    transform = transforms.Compose(
        [
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ]
    )

    model = AgeRegressor().to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    df = pd.read_csv(VAL_CSV)
    df = df[["cache_path", "age_past"]].dropna().copy()
    df["age_past"] = df["age_past"].astype(float)
    loader = DataLoader(CacheDataset(df, transform), batch_size=64, shuffle=False, num_workers=0)

    rows = []
    with torch.no_grad():
        for images, ages, paths in loader:
            images = images.to(device)
            preds = model(images).detach().cpu().numpy()
            for true_age, pred_age, path in zip(ages.numpy(), preds, paths):
                raw = float(max(0, min(100, pred_age)))
                rows.append(
                    {
                        "cache_path": path,
                        "true_age": float(true_age),
                        "pred_age_raw": raw,
                        "error_raw": raw - float(true_age),
                        "abs_error_raw": abs(raw - float(true_age)),
                        "true_group": age_group(true_age),
                        "pred_bin": pred_bin(raw),
                    }
                )

    pred_df = pd.DataFrame(rows)

    bin_stats = (
        pred_df.groupby("pred_bin")
        .agg(
            n=("pred_age_raw", "size"),
            mean_true=("true_age", "mean"),
            mean_pred=("pred_age_raw", "mean"),
            median_residual=("error_raw", lambda x: float(np.median(-x))),
            mae=("abs_error_raw", "mean"),
        )
        .reset_index()
    )
    bin_stats["offset"] = bin_stats["median_residual"].clip(-8, 8)

    offsets = {row["pred_bin"]: round(float(row["offset"]), 3) for _, row in bin_stats.iterrows() if row["n"] >= 20}
    offsets.setdefault("0s", 0.0)
    offsets.setdefault("10s", 0.0)
    offsets.setdefault("20s", 0.0)
    offsets.setdefault("30s", 0.0)
    offsets.setdefault("40s", 0.0)
    offsets.setdefault("50s", 0.0)
    offsets.setdefault("60plus", 0.0)

    pred_df["offset"] = pred_df["pred_bin"].map(offsets).fillna(0.0)
    pred_df["pred_age_calibrated"] = (pred_df["pred_age_raw"] + pred_df["offset"]).clip(0, 100)
    pred_df["error_calibrated"] = pred_df["pred_age_calibrated"] - pred_df["true_age"]
    pred_df["abs_error_calibrated"] = pred_df["error_calibrated"].abs()

    group_report = (
        pred_df.groupby("true_group")
        .agg(
            n=("true_age", "size"),
            true_mean=("true_age", "mean"),
            raw_pred_mean=("pred_age_raw", "mean"),
            calibrated_pred_mean=("pred_age_calibrated", "mean"),
            raw_mae=("abs_error_raw", "mean"),
            calibrated_mae=("abs_error_calibrated", "mean"),
            raw_bias=("error_raw", "mean"),
            calibrated_bias=("error_calibrated", "mean"),
        )
        .reset_index()
    )

    payload = {
        "enabled": True,
        "method": "predicted_age_bin_median_residual",
        "raw_mae": round(float(pred_df["abs_error_raw"].mean()), 4),
        "calibrated_mae": round(float(pred_df["abs_error_calibrated"].mean()), 4),
        "offsets": offsets,
        "bin_stats": bin_stats.to_dict(orient="records"),
    }

    CALIBRATION_PATH.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    group_report.to_csv(GROUP_REPORT_PATH, index=False, encoding="utf-8-sig")
    pred_df.to_csv(PREDICTION_REPORT_PATH, index=False, encoding="utf-8-sig")

    print(f"raw_mae={payload['raw_mae']}")
    print(f"calibrated_mae={payload['calibrated_mae']}")
    print(f"saved {CALIBRATION_PATH}")
    print(f"saved {GROUP_REPORT_PATH}")
    print(f"saved {PREDICTION_REPORT_PATH}")
    print(group_report.to_string(index=False))


if __name__ == "__main__":
    main()
