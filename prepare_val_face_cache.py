from pathlib import Path

from prepare_age_face_cache import build_cache


PROJECT_DIR = Path(r"C:\Users\svc06\OneDrive\Desktop\한국인_이미지_프로젝트")
DATA_DIR = PROJECT_DIR / "data" / "aging_face"


if __name__ == "__main__":
    build_cache(DATA_DIR / "val_dataset.csv", DATA_DIR / "val_face_cache_224.csv", "val")
