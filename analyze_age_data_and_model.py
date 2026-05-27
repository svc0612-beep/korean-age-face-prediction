from pathlib import Path
import json

import numpy as np
import pandas as pd


PROJECT_DIR = Path(r"C:\Users\svc06\OneDrive\Desktop\한국인_이미지_프로젝트")
DATA_DIR = PROJECT_DIR / "data" / "aging_face"
REPORT_DIR = PROJECT_DIR / "reports"

TRAIN_CSV = DATA_DIR / "train_dataset.csv"
VAL_CSV = DATA_DIR / "val_dataset.csv"
TRAIN_CACHE_CSV = DATA_DIR / "train_face_cache_224.csv"
VAL_CACHE_CSV = DATA_DIR / "val_face_cache_224.csv"
HISTORY_CSV = DATA_DIR / "age_mobilenet_history.csv"


def age_group(age):
    if age < 10:
        return "0대"
    if age < 20:
        return "10대"
    if age < 30:
        return "20대"
    if age < 40:
        return "30대"
    if age < 50:
        return "40대"
    if age < 60:
        return "50대"
    return "60대이상"


def path_exists_rate(series):
    exists = series.astype(str).map(lambda p: Path(p).exists())
    return int(exists.sum()), int((~exists).sum())


def summarize_split(name, csv_path):
    df = pd.read_csv(csv_path)
    result = {
        "name": name,
        "rows": int(len(df)),
        "columns": list(df.columns),
        "duplicate_rows": int(df.duplicated().sum()),
        "missing_by_column": {col: int(df[col].isna().sum()) for col in df.columns},
    }

    if "image_path" in df.columns:
        ok, bad = path_exists_rate(df["image_path"])
        result["image_path_exists"] = ok
        result["image_path_missing"] = bad
    if "json_path" in df.columns:
        ok, bad = path_exists_rate(df["json_path"])
        result["json_path_exists"] = ok
        result["json_path_missing"] = bad
    if "cache_path" in df.columns:
        ok, bad = path_exists_rate(df["cache_path"])
        result["cache_path_exists"] = ok
        result["cache_path_missing"] = bad

    if "age_past" in df.columns:
        ages = pd.to_numeric(df["age_past"], errors="coerce")
        result["age_summary"] = {
            "min": float(ages.min()),
            "q1": float(ages.quantile(0.25)),
            "median": float(ages.median()),
            "mean": float(ages.mean()),
            "q3": float(ages.quantile(0.75)),
            "max": float(ages.max()),
            "std": float(ages.std()),
        }
        result["age_invalid_count"] = int(((ages < 0) | (ages > 100) | ages.isna()).sum())
        result["age_group_counts"] = {
            k: int(v)
            for k, v in ages.dropna().map(age_group).value_counts().sort_index().items()
        }

    if "gender" in df.columns:
        result["gender_counts"] = {
            str(k): int(v) for k, v in df["gender"].fillna("missing").str.lower().value_counts().items()
        }
    if "person_id" in df.columns:
        result["person_count"] = int(df["person_id"].nunique())
        result["images_per_person_summary"] = {
            k: float(v)
            for k, v in df["person_id"].value_counts().describe().to_dict().items()
        }
    return result


def summarize_history():
    if not HISTORY_CSV.exists():
        return {"exists": False}
    hist = pd.read_csv(HISTORY_CSV)
    if hist.empty:
        return {"exists": True, "rows": 0}

    best_idx = hist["val_mae"].idxmin()
    best = hist.loc[best_idx]
    last = hist.iloc[-1]
    gap = float(best["val_mae"] - best["train_mae"])

    return {
        "exists": True,
        "rows": int(len(hist)),
        "best_epoch": int(best["epoch"]),
        "best_val_mae": float(best["val_mae"]),
        "best_train_mae": float(best["train_mae"]),
        "best_generalization_gap_val_minus_train": gap,
        "last_epoch": int(last["epoch"]),
        "last_train_mae": float(last["train_mae"]),
        "last_val_mae": float(last["val_mae"]),
        "overfit_signal": bool(gap > 2.0),
    }


def main():
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    report = {
        "raw_train": summarize_split("raw_train", TRAIN_CSV),
        "raw_val": summarize_split("raw_val", VAL_CSV),
        "cache_train": summarize_split("cache_train", TRAIN_CACHE_CSV) if TRAIN_CACHE_CSV.exists() else None,
        "cache_val": summarize_split("cache_val", VAL_CACHE_CSV) if VAL_CACHE_CSV.exists() else None,
        "history": summarize_history(),
        "recommendations": [
            "age_past를 회귀 정답으로 유지한다.",
            "0세와 70세 이상은 이상치가 아니라 실제 라벨일 수 있으므로 제거하지 않고 구간별 MAE로 감시한다.",
            "과적합 신호가 보이면 dropout, weight decay, augmentation, early stopping, 더 많은 train cache를 사용한다.",
            "웹앱 배포 전에는 validation 전체에 대한 구간별 MAE와 예측 샘플을 확인한다.",
        ],
    }

    json_path = REPORT_DIR / "data_quality_report.json"
    md_path = REPORT_DIR / "data_quality_report.md"

    json_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    lines = ["# 데이터 품질 및 모델 진단 리포트", ""]
    for key in ["raw_train", "raw_val", "cache_train", "cache_val"]:
        section = report.get(key)
        if not section:
            continue
        lines += [
            f"## {key}",
            f"- rows: {section.get('rows')}",
            f"- duplicate_rows: {section.get('duplicate_rows')}",
        ]
        if "age_summary" in section:
            lines.append(f"- age_summary: `{section['age_summary']}`")
        if "age_invalid_count" in section:
            lines.append(f"- age_invalid_count: {section['age_invalid_count']}")
        if "age_group_counts" in section:
            lines.append(f"- age_group_counts: `{section['age_group_counts']}`")
        if "gender_counts" in section:
            lines.append(f"- gender_counts: `{section['gender_counts']}`")
        for path_key in ["image_path", "json_path", "cache_path"]:
            exists_key = f"{path_key}_exists"
            missing_key = f"{path_key}_missing"
            if exists_key in section:
                lines.append(f"- {path_key}: exists={section[exists_key]}, missing={section[missing_key]}")
        lines.append("")

    hist = report["history"]
    lines += [
        "## 학습 히스토리",
        f"- best_epoch: {hist.get('best_epoch')}",
        f"- best_val_mae: {hist.get('best_val_mae')}",
        f"- best_train_mae: {hist.get('best_train_mae')}",
        f"- generalization_gap: {hist.get('best_generalization_gap_val_minus_train')}",
        f"- overfit_signal: {hist.get('overfit_signal')}",
        "",
        "## 권장 조치",
    ]
    lines += [f"- {item}" for item in report["recommendations"]]
    md_path.write_text("\n".join(lines), encoding="utf-8")

    print(f"saved {json_path}", flush=True)
    print(f"saved {md_path}", flush=True)


if __name__ == "__main__":
    main()
