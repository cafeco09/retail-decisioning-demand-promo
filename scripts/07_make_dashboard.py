from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

REPORTS = Path("reports")
FIGS = REPORTS / "figures"
FIGS.mkdir(parents=True, exist_ok=True)

def save_fig(path: Path):
    plt.tight_layout()
    plt.savefig(path, dpi=180)
    plt.close()

def main():
    # --- Metrics chart ---
    metrics_path = REPORTS / "metrics.csv"
    if metrics_path.exists():
        m = pd.read_csv(metrics_path).iloc[0].to_dict()
        fig = plt.figure(figsize=(6, 3.2))
        ax = fig.add_subplot(111)
        keys = ["mae", "smape"]
        vals = [m.get("mae", None), m.get("smape", None)]
        ax.bar(keys, vals)
        ax.set_title("Model quality (holdout)")
        ax.set_ylabel("Value")
        for i, v in enumerate(vals):
            if v is not None:
                ax.text(i, v, f"{v:.3f}", ha="center", va="bottom")
        save_fig(FIGS / "01_model_metrics.png")

    # --- Policy comparison ---
    rec_path = REPORTS / "recommended_actions_sample.csv"
    if rec_path.exists():
        r = pd.read_csv(rec_path)

        # Promo rate by policy
        promo_rate = (
            r.assign(promo_on=(r["is_promo"].astype(int) > 0))
             .groupby("policy")["promo_on"]
             .mean()
             .sort_index()
        )

        fig = plt.figure(figsize=(6, 3.2))
        ax = fig.add_subplot(111)
        ax.bar(promo_rate.index, promo_rate.values)
        ax.set_title("Promo usage by policy")
        ax.set_ylabel("Promo rate")
        ax.set_ylim(0, 1)
        for i, v in enumerate(promo_rate.values):
            ax.text(i, v, f"{v:.2f}", ha="center", va="bottom")
        save_fig(FIGS / "02_promo_rate_by_policy.png")

        # Discount distribution by policy (share of decisions)
        r["discount_bucket"] = r["discount_rec"].round(2).astype(str)
        dist = (
            r.groupby(["policy", "discount_bucket"])
             .size()
             .reset_index(name="n")
        )
        totals = dist.groupby("policy")["n"].transform("sum")
        dist["share"] = dist["n"] / totals

        # pivot to plot
        pv = dist.pivot(index="discount_bucket", columns="policy", values="share").fillna(0.0)
        pv = pv.sort_index()

        fig = plt.figure(figsize=(7, 3.6))
        ax = fig.add_subplot(111)
        pv.plot(kind="bar", ax=ax)
        ax.set_title("Discount distribution by policy")
        ax.set_xlabel("Discount (rounded)")
        ax.set_ylabel("Share of decisions")
        ax.set_ylim(0, 1)
        save_fig(FIGS / "03_discount_dist_by_policy.png")

    print(f"Saved figures to: {FIGS}")

if __name__ == "__main__":
    main()
