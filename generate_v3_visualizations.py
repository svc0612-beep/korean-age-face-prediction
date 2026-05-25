from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


PROJECT_DIR = Path(__file__).resolve().parent
DATA_DIR = PROJECT_DIR / "data" / "aging_face"
OUT_DIR = PROJECT_DIR / "assets" / "visualizations"

HISTORY_CSV = DATA_DIR / "age_efficientnet_v3_history.csv"
PRED_CSV = DATA_DIR / "age_efficientnet_v3_val_predictions.csv"

GROUP_ORDER = ["0s", "10s", "20s", "30s", "40s", "50s", "60plus"]


def savefig(name):
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    path = OUT_DIR / name
    plt.tight_layout()
    plt.savefig(path, dpi=180, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"saved {path}", flush=True)


def plot_v3_mae_curve(history):
    fig, ax = plt.subplots(figsize=(10, 5.5))
    ax.plot(history["epoch"], history["train_mae"], label="Train MAE", linewidth=2.2, color="#2563eb")
    ax.plot(history["epoch"], history["val_mae"], label="Validation MAE", linewidth=2.2, color="#f97316")
    ax.plot(history["epoch"], history["adult_mae"], label="Adult MAE", linewidth=2.2, color="#16a34a")
    best_idx = history["score"].idxmin()
    best = history.loc[best_idx]
    ax.scatter([best["epoch"]], [best["val_mae"]], color="#dc2626", zorder=4)
    ax.annotate(
        f"best epoch {int(best['epoch'])}\nval MAE {best['val_mae']:.2f}",
        xy=(best["epoch"], best["val_mae"]),
        xytext=(12, 16),
        textcoords="offset points",
        color="#dc2626",
        fontsize=10,
    )
    ax.set_title("V3 EfficientNet Training MAE", fontsize=17, fontweight="bold")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("MAE (years)")
    ax.grid(alpha=0.25)
    ax.legend(frameon=False)
    savefig("10_v3_mae_curve.png")


def plot_v3_score_group_acc(history):
    fig, ax1 = plt.subplots(figsize=(10, 5.5))
    ax1.plot(history["epoch"], history["score"], color="#7c3aed", linewidth=2.2, label="Score")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Score", color="#7c3aed")
    ax1.tick_params(axis="y", labelcolor="#7c3aed")
    ax1.grid(alpha=0.25)

    ax2 = ax1.twinx()
    ax2.plot(history["epoch"], history["val_group_acc"], color="#0f766e", linewidth=2.2, label="Group Accuracy")
    ax2.set_ylabel("Validation age-group accuracy", color="#0f766e")
    ax2.tick_params(axis="y", labelcolor="#0f766e")

    ax1.set_title("V3 Score and Age-Group Accuracy", fontsize=17, fontweight="bold")
    savefig("11_v3_score_group_acc.png")


def group_summary(preds):
    summary = (
        preds.groupby("true_group")
        .agg(
            n=("true_age", "size"),
            true_mean=("true_age", "mean"),
            pred_mean=("pred_age", "mean"),
            mae=("abs_error", "mean"),
            bias=("bias", "mean"),
        )
        .reindex(GROUP_ORDER)
        .reset_index()
    )
    return summary


def plot_group_mae(summary):
    fig, ax = plt.subplots(figsize=(10, 5.5))
    bars = ax.bar(summary["true_group"], summary["mae"], color="#2563eb")
    ax.bar_label(bars, fmt="%.1f", padding=3, fontsize=10)
    ax.set_title("V3 Validation MAE by True Age Group", fontsize=17, fontweight="bold")
    ax.set_xlabel("True age group")
    ax.set_ylabel("MAE (years)")
    ax.grid(axis="y", alpha=0.25)
    savefig("12_v3_group_mae.png")


def plot_group_bias(summary):
    colors = ["#dc2626" if value < 0 else "#2563eb" for value in summary["bias"]]
    fig, ax = plt.subplots(figsize=(10, 5.5))
    bars = ax.bar(summary["true_group"], summary["bias"], color=colors)
    ax.axhline(0, color="#111827", linewidth=1)
    ax.bar_label(bars, fmt="%.1f", padding=3, fontsize=10)
    ax.set_title("V3 Validation Bias by True Age Group", fontsize=17, fontweight="bold")
    ax.set_xlabel("True age group")
    ax.set_ylabel("Predicted age - true age")
    ax.grid(axis="y", alpha=0.25)
    savefig("13_v3_group_bias.png")


def plot_true_vs_pred(preds):
    sample = preds.sample(n=min(1600, len(preds)), random_state=42)
    fig, ax = plt.subplots(figsize=(7, 7))
    ax.scatter(sample["true_age"], sample["pred_age"], s=14, alpha=0.38, color="#2563eb")
    ax.plot([0, 80], [0, 80], color="#dc2626", linewidth=2, label="Perfect prediction")
    ax.set_xlim(0, 80)
    ax.set_ylim(0, 80)
    ax.set_title("V3 True Age vs Predicted Age", fontsize=17, fontweight="bold")
    ax.set_xlabel("True age")
    ax.set_ylabel("Predicted age")
    ax.grid(alpha=0.25)
    ax.legend(frameon=False)
    savefig("14_v3_true_vs_pred.png")


def plot_error_hist(preds):
    fig, ax = plt.subplots(figsize=(10, 5.5))
    ax.hist(preds["bias"], bins=40, color="#f97316", alpha=0.78)
    ax.axvline(0, color="#111827", linewidth=1.5)
    ax.set_title("V3 Validation Prediction Error Distribution", fontsize=17, fontweight="bold")
    ax.set_xlabel("Predicted age - true age")
    ax.set_ylabel("Count")
    ax.grid(axis="y", alpha=0.25)
    savefig("15_v3_error_hist.png")


def main():
    history = pd.read_csv(HISTORY_CSV)
    preds = pd.read_csv(PRED_CSV)
    summary = group_summary(preds)
    summary.to_csv(OUT_DIR / "v3_group_summary.csv", index=False, encoding="utf-8-sig")

    plot_v3_mae_curve(history)
    plot_v3_score_group_acc(history)
    plot_group_mae(summary)
    plot_group_bias(summary)
    plot_true_vs_pred(preds)
    plot_error_hist(preds)


if __name__ == "__main__":
    main()
