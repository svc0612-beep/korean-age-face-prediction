import base64
import io
import time
from pathlib import Path

from PIL import Image, ImageOps
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models, transforms


BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "data" / "aging_face" / "best_age_efficientnet_v3_multitask.pt"

AGE_GROUPS = ["0s", "10s", "20s", "30s", "40s", "50s", "60plus"]
AGE_GROUP_LABELS = {
    "0s": "0대",
    "10s": "10대",
    "20s": "20대",
    "30s": "30대",
    "40s": "40대",
    "50s": "50대",
    "60plus": "60대 이상",
}
AGE_GROUP_CENTERS = torch.tensor([4.5, 14.5, 24.5, 34.5, 44.5, 54.5, 64.5], dtype=torch.float32)


class AgeEfficientNetMultiTask(nn.Module):
    def __init__(self, num_groups=len(AGE_GROUPS)):
        super().__init__()
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


class AgePredictor:
    def __init__(self, model_path=MODEL_PATH):
        if not Path(model_path).exists():
            raise FileNotFoundError(f"model file not found: {model_path}")

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        checkpoint = torch.load(model_path, map_location=self.device)
        self.model_path = Path(model_path)
        self.model_name = checkpoint.get("model_name", "efficientnet_b0_v3_multitask")
        self.img_size = int(checkpoint.get("img_size", 224))
        self.best_val_mae = float(checkpoint.get("best_val_mae", 4.072))
        self.best_adult_mae = float(checkpoint.get("best_adult_mae", 5.446))
        self.best_epoch = int(checkpoint.get("epoch", 0))
        self.target_mean = float(checkpoint["target_mean"])
        self.target_std = float(checkpoint["target_std"])
        self.age_groups = checkpoint.get("age_groups", AGE_GROUPS)
        self.mean = checkpoint.get("mean", [0.485, 0.456, 0.406])
        self.std = checkpoint.get("std", [0.229, 0.224, 0.225])

        self.model = AgeEfficientNetMultiTask(num_groups=len(self.age_groups)).to(self.device)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.model.eval()

        self.group_centers = AGE_GROUP_CENTERS[: len(self.age_groups)].to(self.device)
        self.transform = transforms.Compose(
            [
                transforms.Resize((self.img_size, self.img_size)),
                transforms.ToTensor(),
                transforms.Normalize(mean=self.mean, std=self.std),
            ]
        )

    @staticmethod
    def webcam_adult_offset(age):
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

    def denormalize_age(self, pred_norm):
        return pred_norm * self.target_std + self.target_mean

    def blend_age_with_group_head(self, regression_age, group_probs):
        group_expected_age = torch.sum(group_probs * self.group_centers, dim=1)
        confidence = torch.max(group_probs, dim=1).values
        blend_weight = torch.clamp((confidence - 0.35) / 0.45, min=0.15, max=0.45)
        blended_age = regression_age * (1.0 - blend_weight) + group_expected_age * blend_weight
        return blended_age, group_expected_age, blend_weight

    @torch.no_grad()
    def predict(self, data_url: str, already_cropped=False, apply_webcam_correction=False):
        start = time.time()
        image = self.image_from_data_url(data_url)
        original_size = image.size
        cropped = image if already_cropped else self.center_face_crop(image)

        tensor = self.transform(cropped).unsqueeze(0).to(self.device)
        pred_norm, group_logits = self.model(tensor)
        regression_age = self.denormalize_age(pred_norm).clamp(0, 100)
        group_probs = F.softmax(group_logits, dim=1)
        blended_age, group_expected_age, blend_weight = self.blend_age_with_group_head(regression_age, group_probs)

        raw_age = float(regression_age.item())
        model_adjusted_age = float(blended_age.clamp(0, 100).item())
        group_expected = float(group_expected_age.item())
        multitask_offset = model_adjusted_age - raw_age
        webcam_offset = self.webcam_adult_offset(model_adjusted_age) if apply_webcam_correction else 0.0
        predicted_age = max(0.0, min(100.0, model_adjusted_age + webcam_offset))
        rounded_age = int(round(predicted_age))
        elapsed_ms = int((time.time() - start) * 1000)

        group_index = int(torch.argmax(group_probs, dim=1).item())
        predicted_group = self.age_groups[group_index]
        predicted_group_label = AGE_GROUP_LABELS.get(predicted_group, predicted_group)
        group_confidence = float(group_probs[0, group_index].item())

        lower = max(0, int(round(predicted_age - self.best_val_mae)))
        upper = min(100, int(round(predicted_age + self.best_val_mae)))
        low_webcam_reliability = predicted_age < 25.0 and original_size[0] == original_size[1]
        reliability_note = (
            "주의: 웹캠 실사용 이미지에서 모델이 나이를 낮게 예측하는 경향이 감지되었습니다."
            if low_webcam_reliability
            else ""
        )
        analysis = [
            "웹캠 입력 이미지를 서버에서 수신했습니다.",
            f"원본 이미지 크기 {original_size[0]}x{original_size[1]}를 확인했습니다.",
            "웹페이지에서 얼굴 가이드 영역을 먼저 정사각형으로 잘라 보냈습니다."
            if already_cropped
            else "얼굴이 중앙에 있다고 가정하고 얼굴 중심 영역을 정사각형으로 잘랐습니다.",
            f"학습 때와 동일하게 {self.img_size}x{self.img_size} 크기로 변환했습니다.",
            "ImageNet 기준 정규화를 적용했습니다.",
            f"학습된 {self.model_name} 파일을 기준으로 추론했습니다.",
            f"나이 회귀 head의 예측값은 {raw_age:.1f}세입니다.",
            f"나이대 분류 head는 {predicted_group_label}를 가장 높게 보았습니다. 신뢰도는 {group_confidence * 100:.1f}%입니다.",
            f"나이대 분류 head의 평균 기대 나이는 {group_expected:.1f}세입니다.",
            f"회귀값과 나이대 판단을 함께 반영한 모델 보정값은 {multitask_offset:+.1f}세입니다.",
        ]
        if apply_webcam_correction:
            analysis.append(f"웹캠 성인 보정 {webcam_offset:+.1f}세를 추가 적용했습니다.")
        if reliability_note:
            analysis.append(reliability_note)
        analysis.append(f"검증 평균 오차는 약 {self.best_val_mae:.2f}세이고, 성인 구간 MAE는 약 {self.best_adult_mae:.2f}세입니다.")

        return {
            "predicted_age": round(predicted_age, 1),
            "raw_age": round(raw_age, 1),
            "model_adjusted_age": round(model_adjusted_age, 1),
            "validation_offset": round(multitask_offset, 1),
            "webcam_offset": round(webcam_offset, 1),
            "rounded_age": rounded_age,
            "age_range": [lower, upper],
            "model_mae": round(self.best_val_mae, 2),
            "adult_mae": round(self.best_adult_mae, 2),
            "best_epoch": self.best_epoch,
            "device": str(self.device),
            "elapsed_ms": elapsed_ms,
            "low_webcam_reliability": low_webcam_reliability,
            "reliability_note": reliability_note,
            "model_name": self.model_name,
            "model_file": self.model_path.name,
            "predicted_group": predicted_group_label,
            "group_confidence": round(group_confidence * 100, 1),
            "group_expected_age": round(group_expected, 1),
            "blend_weight": round(float(blend_weight.item()), 3),
            "analysis": analysis,
        }


predictor = None


def get_predictor():
    global predictor
    if predictor is None:
        predictor = AgePredictor()
    return predictor
