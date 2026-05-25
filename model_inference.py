import base64
import io
import json
import time
from pathlib import Path

from PIL import Image, ImageOps
import torch
import torch.nn as nn
from torchvision import models, transforms


BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "data" / "aging_face" / "best_age_mobilenet_regression.pt"
CALIBRATION_PATH = BASE_DIR / "data" / "aging_face" / "age_calibration.json"


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


class AgePredictor:
    def __init__(self, model_path=MODEL_PATH):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        checkpoint = torch.load(model_path, map_location=self.device)
        self.img_size = int(checkpoint.get("img_size", 224))
        self.best_val_mae = float(checkpoint.get("best_val_mae", 4.052))
        self.best_epoch = int(checkpoint.get("epoch", 0))
        self.mean = checkpoint.get("mean", [0.485, 0.456, 0.406])
        self.std = checkpoint.get("std", [0.229, 0.224, 0.225])

        self.model = AgeRegressor().to(self.device)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.model.eval()
        self.calibration = self.load_calibration()

        self.transform = transforms.Compose(
            [
                transforms.Resize((self.img_size, self.img_size)),
                transforms.ToTensor(),
                transforms.Normalize(mean=self.mean, std=self.std),
            ]
        )

    @staticmethod
    def load_calibration():
        if not CALIBRATION_PATH.exists():
            return {"enabled": False, "offsets": {}}
        try:
            return json.loads(CALIBRATION_PATH.read_text(encoding="utf-8"))
        except Exception:
            return {"enabled": False, "offsets": {}}

    @staticmethod
    def predicted_bin(age):
        age = max(0, min(100, float(age)))
        bucket = int(age // 10) * 10
        return "60plus" if bucket >= 60 else f"{bucket}s"

    def validation_offset(self, age):
        if not self.calibration.get("enabled"):
            return 0.0
        key = self.predicted_bin(age)
        return float(self.calibration.get("offsets", {}).get(key, 0.0))

    @staticmethod
    def webcam_adult_offset(age):
        # Webcam images are usually harsher domain-shifted than AI Hub validation images.
        # Keep this conservative. A strong manual correction can collapse different
        # adult faces into the same age range.
        if age < 18:
            return 8.0
        if age < 25:
            return 5.0
        if age < 30:
            return 3.0
        if age < 40:
            return 1.5
        return 0.0

    @staticmethod
    def image_from_data_url(data_url: str) -> Image.Image:
        if "," in data_url:
            data_url = data_url.split(",", 1)[1]
        image_bytes = base64.b64decode(data_url)
        return Image.open(io.BytesIO(image_bytes)).convert("RGB")

    @staticmethod
    def center_face_crop(image: Image.Image) -> Image.Image:
        image = ImageOps.exif_transpose(image).convert("RGB")
        width, height = image.size
        side = int(min(width, height) * 0.82)
        left = max(0, (width - side) // 2)
        top = max(0, int((height - side) * 0.42))
        right = min(width, left + side)
        bottom = min(height, top + side)
        return image.crop((left, top, right, bottom))

    @torch.no_grad()
    def predict(self, data_url: str, already_cropped=False, apply_webcam_correction=True):
        start = time.time()
        image = self.image_from_data_url(data_url)
        original_size = image.size
        cropped = image if already_cropped else self.center_face_crop(image)

        tensor = self.transform(cropped).unsqueeze(0).to(self.device)
        raw_age = float(self.model(tensor).item())
        raw_age = max(0.0, min(100.0, raw_age))
        validation_offset = self.validation_offset(raw_age)
        webcam_offset = self.webcam_adult_offset(raw_age) if apply_webcam_correction else 0.0
        predicted_age = max(0.0, min(100.0, raw_age + validation_offset + webcam_offset))
        rounded_age = int(round(predicted_age))
        elapsed_ms = int((time.time() - start) * 1000)

        lower = max(0, int(round(predicted_age - self.best_val_mae)))
        upper = min(100, int(round(predicted_age + self.best_val_mae)))
        low_webcam_reliability = raw_age < 25.0 and original_size[0] == original_size[1]
        reliability_note = (
            "\uc8fc\uc758: \uc6f9\ucea0 \uc2e4\uc0ac\uc6a9 \uc774\ubbf8\uc9c0\uc5d0\uc11c \ubaa8\ub378\uc774 \ub098\uc774\ub97c \ub0ae\uac8c \uc608\uce21\ud558\ub294 \uacbd\ud5a5\uc774 \uac10\uc9c0\ub418\uc5c8\uc2b5\ub2c8\ub2e4."
            if low_webcam_reliability
            else ""
        )
        analysis = [
            "\uc6f9\ucea0 \uc785\ub825 \uc774\ubbf8\uc9c0\ub97c \uc11c\ubc84\uc5d0\uc11c \uc218\uc2e0\ud588\uc2b5\ub2c8\ub2e4.",
            f"\uc6d0\ubcf8 \uc774\ubbf8\uc9c0 \ud06c\uae30 {original_size[0]}x{original_size[1]}\ub97c \ud655\uc778\ud588\uc2b5\ub2c8\ub2e4.",
            "\uc6f9\ud398\uc774\uc9c0\uc5d0\uc11c \uc5bc\uad74 \uac00\uc774\ub4dc \uc601\uc5ed\uc744 \uba3c\uc800 \uc815\uc0ac\uac01\ud615\uc73c\ub85c \uc798\ub77c \ubcf4\ub0c8\uc2b5\ub2c8\ub2e4." if already_cropped else "\uc5bc\uad74\uc774 \uc911\uc559\uc5d0 \uc788\ub2e4\uace0 \uac00\uc815\ud558\uace0 \uc5bc\uad74 \uc911\uc2ec \uc601\uc5ed\uc744 \uc815\uc0ac\uac01\ud615\uc73c\ub85c \uc798\ub790\uc2b5\ub2c8\ub2e4.",
            f"\ud559\uc2b5 \ub54c\uc640 \ub3d9\uc77c\ud558\uac8c {self.img_size}x{self.img_size} \ud06c\uae30\ub85c \ubcc0\ud658\ud588\uc2b5\ub2c8\ub2e4.",
            "ImageNet \uae30\uc900 \uc815\uaddc\ud654\ub97c \uc801\uc6a9\ud588\uc2b5\ub2c8\ub2e4.",
            "\ud559\uc2b5\ub41c MobileNetV3 \ud68c\uadc0 \ubaa8\ub378\uc774 age_past \uae30\uc900 \uc608\uc0c1 \ub098\uc774\ub97c \uacc4\uc0b0\ud588\uc2b5\ub2c8\ub2e4.",
            f"\ubcf4\uc815 \uc804 \uc608\uce21\uac12\uc740 {raw_age:.1f}\uc138\uc785\ub2c8\ub2e4.",
            f"\uac80\uc99d \ud1b5\uacc4 \ubcf4\uc815 {validation_offset:+.1f}\uc138, \uc6f9\ucea0 \uc131\uc778 \ubcf4\uc815 {webcam_offset:+.1f}\uc138\ub97c \uc801\uc6a9\ud588\uc2b5\ub2c8\ub2e4.",
        ]
        if reliability_note:
            analysis.append(reliability_note)
        analysis.append(f"\uac80\uc99d \ud3c9\uade0 \uc624\ucc28\ub294 \uc57d {self.best_val_mae:.2f}\uc138\uc785\ub2c8\ub2e4.")

        return {
            "predicted_age": round(predicted_age, 1),
            "raw_age": round(raw_age, 1),
            "validation_offset": round(validation_offset, 1),
            "webcam_offset": round(webcam_offset, 1),
            "rounded_age": rounded_age,
            "age_range": [lower, upper],
            "model_mae": round(self.best_val_mae, 2),
            "best_epoch": self.best_epoch,
            "device": str(self.device),
            "elapsed_ms": elapsed_ms,
            "low_webcam_reliability": low_webcam_reliability,
            "reliability_note": reliability_note,
            "analysis": analysis,
        }


predictor = None


def get_predictor():
    global predictor
    if predictor is None:
        predictor = AgePredictor()
    return predictor
