"""
analysis.py  -  all the calculations for the dashboard.
It repeats the same steps as our Colab notebook, so the numbers match.
"""
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

# The four price ranges (price_range value -> name) and their chart colours
TIERS = {0: "Low cost", 1: "Medium", 2: "High", 3: "Very high"}
COLORS = {"Low cost": "#38bdf8", "Medium": "#8b5cf6", "High": "#f4b942", "Very high": "#f4697b"}
SPECS = {
    "ram": "RAM (MB)",
    "battery_power": "Battery (mAh)",
    "int_memory": "Internal memory (GB)",
    "px_width": "Pixel width",
    "px_height": "Pixel height",
    "mobile_wt": "Weight (g)",
}


def load_raw(path):
    """Read the raw Kaggle file, train.csv (notebook cell 2)."""
    return pd.read_csv(path)


def clean(df):
    """Cleaning steps from the notebook (cells 6 to 9): drop the rows where px_height is 0,
    and replace the zero screen widths (sc_w) with the median width."""
    log = {"rows_before": len(df),
           "px_height_zero": int((df["px_height"] == 0).sum()),
           "sc_w_zero": int((df["sc_w"] == 0).sum())}
    df = df[df["px_height"] > 0].copy()
    df["sc_w"] = df["sc_w"].replace(0, np.nan)
    df["sc_w"] = df["sc_w"].fillna(df["sc_w"].median())
    log["rows_after"] = len(df)
    return df, log


def load_data(path):
    """Raw file in, analysis-ready table out (cleaned, with a price tier name column)."""
    df, _ = clean(load_raw(path))
    df["tier"] = df["price_range"].map(TIERS)
    return df


def fp(p):
    return "< 0.001" if p < 0.001 else f"{p:.3f}"


def run_tests(d):
    g = lambda c: [d[d["price_range"] == k][c] for k in range(4)]
    f_ram, p_ram = stats.f_oneway(*g("ram"))
    f_bat, p_bat = stats.f_oneway(*g("battery_power"))
    rho, p_mem = stats.spearmanr(d["int_memory"], d["price_range"])
    t, p_t = stats.ttest_ind(g("battery_power")[0], g("battery_power")[3], equal_var=False)
    return dict(f_ram=f_ram, p_ram=p_ram, f_bat=f_bat, p_bat=p_bat, rho=rho, p_mem=p_mem, p_t=p_t)


def train_models(_d):
    X = _d.drop(columns=["price_range", "tier"])
    y = _d["price_range"]
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    models = {
        "Logistic Regression": make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000)),
        "Decision Tree": DecisionTreeClassifier(random_state=42, max_depth=6),
        "Random Forest": RandomForestClassifier(random_state=42, n_estimators=100),
    }
    rows = []
    for name, m in models.items():
        m.fit(Xtr, ytr)
        k = int((m.predict(Xte) == yte).sum())
        rows.append({"Model": name, "Test accuracy (%)": round(k / len(yte) * 100, 1), "correct": k, "n": len(yte)})
    return models, pd.DataFrame(rows), list(X.columns), X.median()