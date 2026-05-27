from pathlib import Path
import json

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import font_manager, rcParams


PROJECT_DIR = Path(r"C:\Users\svc06\OneDrive\Desktop\한국인_이미지_프로젝트")
DATA_DIR = PROJECT_DIR / "data" / "aging_face"
OUTPUT_DIR = PROJECT_DIR / "assets" / "visualizations"

TRAIN_CSV = DATA_DIR / "train_dataset.csv"
VAL_CSV = DATA_DIR / "val_dataset.csv"
HISTORY_CSV = DATA_DIR / "age_mobilenet_history.csv"

AGE_ORDER = ["0대", "10대", "20대", "30대", "40대", "50대", "60대이상"]
GENDER_ORDER = ["female", "male"]


def setup_korean_font():
    candidates = ["Malgun Gothic", "AppleGothic", "NanumGothic", "Noto Sans CJK KR"]
    available = {f.name for f in font_manager.fontManager.ttflist}
    for name in candidates:
        if name in available:
            rcParams["font.family"] = name
            break
    rcParams["axes.unicode_minus"] = False
    rcParams["figure.facecolor"] = "white"
    rcParams["axes.facecolor"] = "white"


def savefig(name: str):
    path = OUTPUT_DIR / name
    plt.tight_layout()
    plt.savefig(path, dpi=180, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"saved {path}", flush=True)


def load_data():
    train_df = pd.read_csv(TRAIN_CSV)
    val_df = pd.read_csv(VAL_CSV)
    train_df["split"] = "train"
    val_df["split"] = "val"
    df = pd.concat([train_df, val_df], ignore_index=True)

    df["age_past"] = pd.to_numeric(df["age_past"], errors="coerce")
    df = df.dropna(subset=["age_past"]).copy()
    df["age_past"] = df["age_past"].astype(float)

    if "age_group" not in df.columns:
        bins = [0, 10, 20, 30, 40, 50, 60, np.inf]
        df["age_group"] = pd.cut(
            df["age_past"],
            bins=bins,
            labels=AGE_ORDER,
            right=False,
            include_lowest=True,
        )

    df["age_group"] = pd.Categorical(df["age_group"], categories=AGE_ORDER, ordered=True)
    if "gender" in df.columns:
        df["gender"] = df["gender"].fillna("unknown").astype(str).str.lower()
    return train_df, val_df, df


def plot_age_group_distribution(df):
    counts = (
        df.groupby(["age_group", "split"], observed=False)
        .size()
        .unstack(fill_value=0)
        .reindex(AGE_ORDER)
    )
    counts = counts.reindex(columns=["train", "val"], fill_value=0)

    x = np.arange(len(AGE_ORDER))
    width = 0.38
    fig, ax = plt.subplots(figsize=(10, 5.5))
    train_bars = ax.bar(x - width / 2, counts["train"], width, label="Train", color="#2563eb")
    val_bars = ax.bar(x + width / 2, counts["val"], width, label="Validation", color="#f97316")

    ax.set_title("나이대 분포", fontsize=16, fontweight="bold")
    ax.set_xlabel("나이대")
    ax.set_ylabel("이미지 수")
    ax.set_xticks(x)
    ax.set_xticklabels(AGE_ORDER)
    ax.legend(frameon=False)
    ax.grid(axis="y", alpha=0.25)
    ax.bar_label(train_bars, padding=3, fontsize=8)
    ax.bar_label(val_bars, padding=3, fontsize=8)
    savefig("01_age_group_distribution.png")


def plot_gender_donut(df):
    counts = df["gender"].value_counts().reindex(GENDER_ORDER).dropna()
    colors = ["#ef476f", "#118ab2"]

    fig, ax = plt.subplots(figsize=(7, 7))
    wedges, texts, autotexts = ax.pie(
        counts.values,
        labels=counts.index,
        autopct=lambda pct: f"{pct:.1f}%",
        startangle=90,
        colors=colors[: len(counts)],
        wedgeprops={"width": 0.42, "edgecolor": "white"},
        textprops={"fontsize": 12},
    )
    ax.set_title("성별 분포", fontsize=16, fontweight="bold")
    ax.text(0, 0, f"총 {counts.sum():,}장", ha="center", va="center", fontsize=14, fontweight="bold")
    savefig("02_gender_distribution_donut.png")


def plot_split_donut(df):
    counts = df["split"].value_counts().reindex(["train", "val"])
    colors = ["#2563eb", "#f97316"]

    fig, ax = plt.subplots(figsize=(7, 7))
    ax.pie(
        counts.values,
        labels=counts.index,
        autopct=lambda pct: f"{pct:.1f}%",
        startangle=90,
        colors=colors,
        wedgeprops={"width": 0.42, "edgecolor": "white"},
        textprops={"fontsize": 12},
    )
    ax.set_title("Train / Validation 분포", fontsize=16, fontweight="bold")
    ax.text(0, 0, f"총 {counts.sum():,}장", ha="center", va="center", fontsize=14, fontweight="bold")
    savefig("03_split_distribution_donut.png")


def plot_age_histogram(df):
    fig, ax = plt.subplots(figsize=(10, 5.5))
    for split, color in [("train", "#2563eb"), ("val", "#f97316")]:
        values = df.loc[df["split"] == split, "age_past"]
        ax.hist(values, bins=range(0, 86, 2), alpha=0.55, label=split, color=color)

    ax.set_title("실제 나이(age_past) 히스토그램", fontsize=16, fontweight="bold")
    ax.set_xlabel("사진 속 당시 나이")
    ax.set_ylabel("이미지 수")
    ax.legend(frameon=False)
    ax.grid(axis="y", alpha=0.25)
    savefig("04_age_past_histogram.png")


def plot_age_by_gender_boxplot(df):
    groups = []
    labels = []
    colors = []
    for split in ["train", "val"]:
        for gender in GENDER_ORDER:
            values = df.loc[(df["split"] == split) & (df["gender"] == gender), "age_past"].dropna()
            if len(values):
                groups.append(values)
                labels.append(f"{split}\n{gender}")
                colors.append("#ef476f" if gender == "female" else "#118ab2")

    fig, ax = plt.subplots(figsize=(9, 5.5))
    box = ax.boxplot(groups, tick_labels=labels, patch_artist=True, showfliers=False)
    for patch, color in zip(box["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.75)
    ax.set_title("성별/분할별 나이 분포", fontsize=16, fontweight="bold")
    ax.set_ylabel("age_past")
    ax.grid(axis="y", alpha=0.25)
    savefig("05_age_by_gender_boxplot.png")


def plot_gender_age_heatmap(df):
    pivot = (
        df.groupby(["gender", "age_group"], observed=False)
        .size()
        .unstack(fill_value=0)
        .reindex(index=GENDER_ORDER)
        .reindex(columns=AGE_ORDER)
    )

    fig, ax = plt.subplots(figsize=(10, 4.5))
    im = ax.imshow(pivot.values, cmap="YlGnBu")
    ax.set_title("성별 x 나이대 이미지 수", fontsize=16, fontweight="bold")
    ax.set_xticks(np.arange(len(AGE_ORDER)))
    ax.set_yticks(np.arange(len(pivot.index)))
    ax.set_xticklabels(AGE_ORDER)
    ax.set_yticklabels(pivot.index)

    for i in range(pivot.shape[0]):
        for j in range(pivot.shape[1]):
            ax.text(j, i, f"{pivot.values[i, j]:,}", ha="center", va="center", fontsize=9)

    fig.colorbar(im, ax=ax, fraction=0.025, pad=0.02)
    savefig("06_gender_age_heatmap.png")


def plot_training_history():
    if not HISTORY_CSV.exists():
        print(f"skip training history: {HISTORY_CSV} not found", flush=True)
        return

    hist = pd.read_csv(HISTORY_CSV)
    if hist.empty:
        return

    best_idx = hist["val_mae"].idxmin()
    best_row = hist.loc[best_idx]

    fig, ax = plt.subplots(figsize=(10, 5.5))
    ax.plot(hist["epoch"], hist["train_mae"], label="Train MAE", color="#2563eb", linewidth=2)
    ax.plot(hist["epoch"], hist["val_mae"], label="Validation MAE", color="#f97316", linewidth=2)
    ax.scatter([best_row["epoch"]], [best_row["val_mae"]], color="#dc2626", zorder=5)
    ax.annotate(
        f"best {best_row['val_mae']:.2f}세",
        xy=(best_row["epoch"], best_row["val_mae"]),
        xytext=(8, 12),
        textcoords="offset points",
        fontsize=10,
        color="#dc2626",
    )
    ax.set_title("학습 MAE 변화", fontsize=16, fontweight="bold")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("평균 절대 오차(세)")
    ax.legend(frameon=False)
    ax.grid(alpha=0.25)
    savefig("07_training_mae_curve.png")

    fig, ax = plt.subplots(figsize=(10, 5.5))
    ax.plot(hist["epoch"], hist["train_loss"], label="Train Loss", color="#2563eb", linewidth=2)
    ax.plot(hist["epoch"], hist["val_loss"], label="Validation Loss", color="#f97316", linewidth=2)
    ax.set_title("학습 Loss 변화", fontsize=16, fontweight="bold")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Smooth L1 Loss")
    ax.legend(frameon=False)
    ax.grid(alpha=0.25)
    savefig("08_training_loss_curve.png")

    fig, ax = plt.subplots(figsize=(10, 4.8))
    ax.plot(hist["epoch"], hist["lr"], color="#7c3aed", linewidth=2)
    ax.set_title("Learning Rate 변화", fontsize=16, fontweight="bold")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Learning Rate")
    ax.set_yscale("log")
    ax.grid(alpha=0.25)
    savefig("09_learning_rate_curve.png")


def write_summary(train_df, val_df, df):
    summary = {
        "total_images": int(len(df)),
        "train_images": int(len(train_df)),
        "val_images": int(len(val_df)),
        "age_min": float(df["age_past"].min()),
        "age_max": float(df["age_past"].max()),
        "age_mean": float(df["age_past"].mean()),
        "age_median": float(df["age_past"].median()),
        "age_group_counts": {
            str(k): int(v)
            for k, v in df["age_group"].value_counts().reindex(AGE_ORDER, fill_value=0).items()
        },
        "gender_counts": {str(k): int(v) for k, v in df["gender"].value_counts().items()},
    }

    if HISTORY_CSV.exists():
        hist = pd.read_csv(HISTORY_CSV)
        if not hist.empty:
            best = hist.loc[hist["val_mae"].idxmin()]
            summary["best_epoch"] = int(best["epoch"])
            summary["best_val_mae"] = float(best["val_mae"])

    (OUTPUT_DIR / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    pd.DataFrame([summary]).to_csv(OUTPUT_DIR / "summary.csv", index=False, encoding="utf-8-sig")
    print(f"saved {OUTPUT_DIR / 'summary.json'}", flush=True)
    print(f"saved {OUTPUT_DIR / 'summary.csv'}", flush=True)


def main():
    setup_korean_font()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    train_df, val_df, df = load_data()
    plot_age_group_distribution(df)
    plot_gender_donut(df)
    plot_split_donut(df)
    plot_age_histogram(df)
    plot_age_by_gender_boxplot(df)
    plot_gender_age_heatmap(df)
    plot_training_history()
    write_summary(train_df, val_df, df)


if __name__ == "__main__":
    main()
