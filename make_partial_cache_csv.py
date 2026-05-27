from pathlib import Path

import pandas as pd


PROJECT_DIR = Path(r"C:\Users\svc06\OneDrive\Desktop\한국인_이미지_프로젝트")
DATA_DIR = PROJECT_DIR / "data" / "aging_face"
CACHE_DIR = DATA_DIR / "face_cache_224"


def attach_existing_cache(source_csv: Path, output_csv: Path, split: str):
    df = pd.read_csv(source_csv)
    cache_paths = []
    keep = []
    for _, row in df.iterrows():
        filename_key = str(row.get("filename_key") or Path(row["image_path"]).stem)
        cache_path = CACHE_DIR / split / f"{filename_key}.jpg"
        exists = cache_path.exists()
        keep.append(exists)
        cache_paths.append(str(cache_path))

    out = df.loc[keep].copy()
    out["cache_path"] = [p for p, ok in zip(cache_paths, keep) if ok]
    out.to_csv(output_csv, index=False, encoding="utf-8-sig")
    print(f"{split}: {len(out)}/{len(df)} saved -> {output_csv}", flush=True)


def main():
    attach_existing_cache(DATA_DIR / "train_dataset.csv", DATA_DIR / "train_face_cache_224.csv", "train")
    attach_existing_cache(DATA_DIR / "val_dataset.csv", DATA_DIR / "val_face_cache_224.csv", "val")


if __name__ == "__main__":
    main()
