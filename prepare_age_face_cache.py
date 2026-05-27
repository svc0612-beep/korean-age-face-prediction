from pathlib import Path
import json
import time

import pandas as pd
from PIL import Image, ImageOps


PROJECT_DIR = Path(r"C:\Users\svc06\OneDrive\Desktop\한국인_이미지_프로젝트")
DATA_DIR = PROJECT_DIR / "data" / "aging_face"
TRAIN_CSV = DATA_DIR / "train_dataset.csv"
VAL_CSV = DATA_DIR / "val_dataset.csv"
CACHE_DIR = DATA_DIR / "face_cache_224"
TRAIN_CACHE_CSV = DATA_DIR / "train_face_cache_224.csv"
VAL_CACHE_CSV = DATA_DIR / "val_face_cache_224.csv"

IMG_SIZE = 224
MARGIN = 0.35
JPEG_QUALITY = 92
LOG_EVERY = 500


def read_bbox(json_path: Path):
    data = json.loads(json_path.read_text(encoding="utf-8"))
    annotation = data.get("annotation") or []
    if not annotation:
        return None
    box = annotation[0].get("box")
    if not box:
        return None
    return float(box["x"]), float(box["y"]), float(box["w"]), float(box["h"])


def expanded_crop_box(box, width, height):
    x, y, w, h = box
    cx = x + w / 2
    cy = y + h / 2
    side = max(w, h) * (1 + MARGIN)
    left = int(round(cx - side / 2))
    top = int(round(cy - side / 2))
    right = int(round(cx + side / 2))
    bottom = int(round(cy + side / 2))
    return max(0, left), max(0, top), min(width, right), min(height, bottom)


def cache_one(row, split: str):
    image_path = Path(row["image_path"])
    json_path = Path(row["json_path"])
    filename_key = str(row.get("filename_key") or image_path.stem)
    cache_path = CACHE_DIR / split / f"{filename_key}.jpg"

    if cache_path.exists():
        return str(cache_path)

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    image = Image.open(image_path).convert("RGB")
    image = ImageOps.exif_transpose(image)

    try:
        box = read_bbox(json_path)
    except Exception:
        box = None

    if box is not None:
        crop_box = expanded_crop_box(box, image.width, image.height)
        if crop_box[2] > crop_box[0] and crop_box[3] > crop_box[1]:
            image = image.crop(crop_box)

    image = ImageOps.fit(image, (IMG_SIZE, IMG_SIZE), method=Image.Resampling.LANCZOS)
    image.save(cache_path, "JPEG", quality=JPEG_QUALITY, optimize=True)
    return str(cache_path)


def build_cache(csv_path: Path, out_csv_path: Path, split: str):
    df = pd.read_csv(csv_path)
    required = ["image_path", "json_path", "filename_key", "age_past"]
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise ValueError(f"{csv_path} missing columns: {missing}")

    cached_paths = []
    start = time.time()
    for idx, row in df.iterrows():
        cached_paths.append(cache_one(row, split))
        done = idx + 1
        if done == 1 or done % LOG_EVERY == 0 or done == len(df):
            elapsed = time.time() - start
            rate = done / elapsed if elapsed else 0
            print(f"{split} cache {done}/{len(df)} rate={rate:.1f}/s", flush=True)

    df = df.copy()
    df["cache_path"] = cached_paths
    out_csv_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_csv_path, index=False, encoding="utf-8-sig")
    print(f"saved {out_csv_path}", flush=True)


def main():
    print(f"cache_dir={CACHE_DIR}", flush=True)
    build_cache(TRAIN_CSV, TRAIN_CACHE_CSV, "train")
    build_cache(VAL_CSV, VAL_CACHE_CSV, "val")


if __name__ == "__main__":
    main()
