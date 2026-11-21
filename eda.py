"""EDA with visualizations -> figures/*.png + figures/eda_summary.txt"""
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

import cslib
import model

OUT = model.FIG_DIR


def run(data_dir=None):
    os.makedirs(OUT, exist_ok=True)
    df = cslib.fetch_data(data_dir or model.DEFAULT_DATA_DIR)
    ts = cslib.convert_to_ts(df)
    top = cslib.top_countries(df, 10)

    fig, ax = plt.subplots(figsize=(11, 4))
    ax.plot(ts.index, ts["revenue"], alpha=.35, label="daily")
    ax.plot(ts.index, ts["revenue"].rolling(30).mean(), lw=2, label="30-day mean")
    ax.set(title="Daily revenue, all countries", ylabel="revenue"); ax.legend()
    fig.tight_layout(); fig.savefig(f"{OUT}/eda_revenue_over_time.png", dpi=120); plt.close(fig)

    rev = df.groupby("country_key")["price"].sum().sort_values(ascending=False).head(10)
    fig, ax = plt.subplots(figsize=(8, 4.5)); rev[::-1].plot.barh(ax=ax, color="#2b6cb0")
    ax.set(title="Total revenue by country (top 10)", xlabel="revenue", ylabel="")
    fig.tight_layout(); fig.savefig(f"{OUT}/eda_revenue_by_country.png", dpi=120); plt.close(fig)

    dow = ts["revenue"].groupby(ts.index.dayofweek).mean()
    dow.index = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
    fig, ax = plt.subplots(figsize=(7, 4)); dow.plot.bar(ax=ax, color="#2b6cb0")
    ax.set(title="Mean daily revenue by weekday", ylabel="revenue")
    fig.tight_layout(); fig.savefig(f"{OUT}/eda_weekday_pattern.png", dpi=120); plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 4)); sns.histplot(df["price"], bins=60, log_scale=True, ax=ax)
    ax.set(title="Distribution of invoice-line value (log x)")
    fig.tight_layout(); fig.savefig(f"{OUT}/eda_price_distribution.png", dpi=120); plt.close(fig)

    X, y = cslib.build_xy(ts)
    corr = X.assign(target_30d=y).corr()
    fig, ax = plt.subplots(figsize=(8, 6)); sns.heatmap(corr, annot=True, fmt=".2f", cmap="Blues", ax=ax)
    ax.set_title("Feature / target correlation (all countries)")
    fig.tight_layout(); fig.savefig(f"{OUT}/eda_feature_correlation.png", dpi=120); plt.close(fig)

    share = rev / df["price"].sum()
    lines = [f"rows: {len(df):,}   date range: {df['date'].min().date()} -> {df['date'].max().date()}",
             f"countries: {df['country_key'].nunique()}   top-10 share of revenue: {share.sum():.1%}",
             f"largest market: {rev.index[0]} ({share.iloc[0]:.1%})",
             f"weekday low: {dow.idxmin()}  weekday high: {dow.idxmax()}  (ratio {dow.max()/dow.min():.2f})",
             f"trailing-28-day revenue vs next-30-day revenue correlation: {corr.loc['past_28','target_30d']:.2f}",
             "=> strong persistence + weekly seasonality: rolling-window features are a sensible design,",
             "   and a run-rate baseline will be hard to beat by a large margin."]
    open(f"{OUT}/eda_summary.txt", "w").write("\n".join(lines))
    print("\n".join(lines))


if __name__ == "__main__":
    run(sys.argv[1] if len(sys.argv) > 1 else None)
