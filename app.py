"""
Mobile Phone Price Range dashboard (Streamlit).
Run it with:   streamlit run app.py
The calculations live in analysis.py, this file only shows them.
"""
from pathlib import Path

import streamlit as st
import pandas as pd
import plotly.express as px
from scipy import stats

from analysis import TIERS, COLORS, SPECS, fp, load_data
from analysis import run_tests as calc_tests, train_models as calc_models

st.set_page_config(page_title="Mobile Price Dashboard", page_icon="📱", layout="wide")



st.markdown("""
<style>
.stApp{background:#f6f7fb}
header[data-testid="stHeader"]{background:transparent}
.block-container{padding-top:2rem;max-width:1200px}
section[data-testid="stSidebar"]{background:#0f172a}
section[data-testid="stSidebar"] p, section[data-testid="stSidebar"] label p{color:#cbd5e1}
.logo{font-size:22px;font-weight:700;color:#fff;margin-bottom:18px}
.logo span{color:#00b4ff}
.eyebrow{font-size:12px;font-weight:600;letter-spacing:.08em;color:#0284c7;text-transform:uppercase}
.ttl{font-size:34px;font-weight:700;color:#0f172a;margin:2px 0}
.sub{color:#64748b;margin-bottom:18px}
.kpi{background:#fff;border:1px solid #e5e7eb;border-radius:14px;padding:16px 18px;box-shadow:0 1px 3px rgba(15,23,42,.06)}
.kl{color:#64748b;font-size:13px}
.kv{color:#0f172a;font-size:28px;font-weight:700;margin:4px 0}
.kn{color:#10b981;font-size:12px;font-weight:600}
.ct{font-size:17px;font-weight:600;color:#0f172a}
.cs{font-size:12px;color:#94a3b8;margin-bottom:6px}
.tk{background:#f8fafc;border-radius:10px;padding:12px 14px;margin-bottom:10px}
.tk b{color:#0f172a;font-size:14px}
.tk span{color:#64748b;font-size:13px}
.bar-row{display:flex;align-items:center;gap:12px;margin:10px 0}
.bn{width:160px;color:#0f172a;font-size:14px}
.bt{flex:1;background:#eef2f7;border-radius:8px;height:10px}
.bf{height:10px;border-radius:8px;background:linear-gradient(90deg,#00b4ff,#8b5cf6)}
.bv{width:40px;text-align:right;color:#64748b;font-size:13px}
div[data-testid="stVerticalBlockBorderWrapper"]{background:#fff;border-radius:14px}
[data-testid="stMain"] label p, [data-testid="stWidgetLabel"] p{color:#334155 !important}
section[data-testid="stSidebar"] [data-testid="stWidgetLabel"] p, section[data-testid="stSidebar"] label p{color:#cbd5e1 !important}
[data-testid="stMain"] .stDownloadButton button, [data-testid="stMain"] [data-testid="stDownloadButton"] button{background:#0f172a !important;border:1px solid #0f172a !important;border-radius:10px !important}
[data-testid="stMain"] .stDownloadButton button p, [data-testid="stMain"] [data-testid="stDownloadButton"] button p, [data-testid="stMain"] .stDownloadButton button span{color:#ffffff !important}
[data-testid="stMain"] .stDownloadButton button:hover, [data-testid="stMain"] .stDownloadButton button:focus, [data-testid="stMain"] .stDownloadButton button:active, [data-testid="stMain"] [data-testid="stDownloadButton"] button:hover{background:#0284c7 !important;border-color:#0284c7 !important}
[data-testid="stMain"] .stDownloadButton button:hover p, [data-testid="stMain"] .stDownloadButton button:focus p, [data-testid="stMain"] [data-testid="stDownloadButton"] button:hover p{color:#ffffff !important}
section[data-testid="stSidebar"] button p, section[data-testid="stSidebar"] button span{color:#0f172a !important}
</style>
""", unsafe_allow_html=True)


@st.cache_data
def get_data():
    """Read train.csv once, clean it (analysis.py) and remember it, so the page stays fast."""
    return load_data(Path(__file__).parent / "train.csv")


@st.cache_data
def run_tests(d):
    return calc_tests(d)


@st.cache_resource
def train_models(_d):
    return calc_models(_d)


data = get_data()
num_cols = [c for c in data.columns if c != "tier"]
corr = data[num_cols].corr()
price_corr = corr["price_range"].drop("price_range")


def header(eyebrow, title, sub):
    st.markdown(f'<div class="eyebrow">{eyebrow}</div><div class="ttl">{title}</div><div class="sub">{sub}</div>',
                unsafe_allow_html=True)


def kpi(label, value, note):
    return f'<div class="kpi"><div class="kl">{label}</div><div class="kv">{value}</div><div class="kn">{note}</div></div>'


def card_title(title, sub):
    st.markdown(f'<div class="ct">{title}</div><div class="cs">{sub}</div>', unsafe_allow_html=True)


def style(fig, h=320, legend=False):
    fig.update_layout(template="plotly_white", height=h, margin=dict(l=10, r=10, t=10, b=10),
                      paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", showlegend=legend)
    return fig


# ---------------- Sidebar ----------------
with st.sidebar:
    st.markdown('<div class="logo">📱 Price<span>scope</span></div>', unsafe_allow_html=True)
    page = st.radio("Navigate", ["Overview", "Specification trends", "Phone explorer", "Findings and insights", "Price predictor", "Reports", "User guide"],
                    label_visibility="collapsed")

for t in TIERS.values():
    st.session_state.setdefault("tier_" + t, True)


def filter_bar():
    with st.container(border=True):
        fc = st.columns([1.3, 1, 1, 1, 1, 1.3])
        fc[0].markdown("**Show price tiers:**")
        for col, t in zip(fc[1:5], TIERS.values()):
            col.checkbox(t, key="tier_" + t)
        fc[5].button("Reset filters", on_click=lambda: st.session_state.update({"tier_" + t: True for t in TIERS.values()}))
    chosen = [t for t in TIERS.values() if st.session_state["tier_" + t]]
    if not chosen:
        st.warning("Tick at least one price tier in the bar above.")
        st.stop()
    return data[data["tier"].isin(chosen)], chosen


df = data
order = list(TIERS.values())

# ---------------- Overview ----------------
if page == "Overview":
    header("Live analysis", "Mobile price intelligence", "What phone specifications say about price tier.")
    df, order = filter_bar()
    k = st.columns(4)
    k[0].markdown(kpi("Total phones", f"{len(df):,}", "after cleaning"), unsafe_allow_html=True)
    k[1].markdown(kpi("Average RAM", f"{df['ram'].mean():,.0f} MB", "selected tiers"), unsafe_allow_html=True)
    k[2].markdown(kpi("Average battery", f"{df['battery_power'].mean():,.0f} mAh", "selected tiers"), unsafe_allow_html=True)
    k[3].markdown(kpi("Price tiers", f"{df['tier'].nunique()} of 4", "ticked above the charts"), unsafe_allow_html=True)
    st.write("")

    left, right = st.columns([3, 2])
    with left:
        with st.container(border=True):
            card_title("Average specification by price tier", "Pick a specification")
            spec = st.selectbox("Specification", list(SPECS), format_func=SPECS.get, key="ov_spec")
            avg = df.groupby("tier")[spec].mean().reindex(order).reset_index()
            fig = px.bar(avg, x="tier", y=spec, color="tier", color_discrete_map=COLORS,
                         labels={"tier": "Price tier", spec: SPECS[spec]})
            st.plotly_chart(style(fig), use_container_width=True, theme=None)
    with right:
        with st.container(border=True):
            card_title("Price tier mix", "Share of phones")
            counts = df.groupby("tier").size().reset_index(name="phones")
            fig = px.pie(counts, names="tier", values="phones", hole=0.62,
                         color="tier", color_discrete_map=COLORS)
            st.plotly_chart(style(fig, 360, True), use_container_width=True, theme=None)

    left, right = st.columns([3, 2])
    with left:
        with st.container(border=True):
            card_title("Specification comparison", "Spread of the chosen specification in each tier")
            fig = px.box(df, x="tier", y=spec, color="tier", color_discrete_map=COLORS,
                         category_orders={"tier": order}, labels={"tier": "Price tier", spec: SPECS[spec]})
            st.plotly_chart(style(fig), use_container_width=True, theme=None)
    with right:
        with st.container(border=True):
            card_title("Key takeaways", "Calculated from all phones")
            st.markdown(
                f'<div class="tk"><b>RAM is the strongest signal</b><br><span>Correlation with price tier is {price_corr["ram"]:.2f}; average RAM rises at every tier.</span></div>'
                f'<div class="tk"><b>Battery helps, but less</b><br><span>Correlation with price tier is {price_corr["battery_power"]:.2f}.</span></div>'
                f'<div class="tk"><b>Internal memory barely matters</b><br><span>Correlation with price tier is {price_corr["int_memory"]:.2f}, close to zero.</span></div>',
                unsafe_allow_html=True)

    with st.container(border=True):
        card_title("Feature ranking", "Absolute correlation with price tier (all phones)")
        rank = price_corr.abs().sort_values(ascending=False).head(8)
        bars = "".join(
            f'<div class="bar-row"><span class="bn">{SPECS.get(n, n)}</span>'
            f'<div class="bt"><div class="bf" style="width:{v / rank.max() * 100:.0f}%"></div></div>'
            f'<span class="bv">{v:.2f}</span></div>' for n, v in rank.items())
        st.markdown(bars, unsafe_allow_html=True)

# ---------------- Specification trends ----------------
elif page == "Specification trends":
    header("Trend intelligence", "Specification trends", "How specifications are spread across the price tiers.")
    df, order = filter_bar()
    with st.container(border=True):
        card_title("Distribution of a specification", "How many phones of each tier fall at each value")
        spec = st.selectbox("Specification", list(SPECS), format_func=SPECS.get, key="dist_spec")
        fig = px.histogram(df, x=spec, color="tier", barmode="overlay", opacity=0.65, nbins=30,
                           color_discrete_map=COLORS, category_orders={"tier": order},
                           labels={spec: SPECS[spec], "tier": "Price tier"})
        fig.update_layout(yaxis_title="Number of phones")
        st.plotly_chart(style(fig, 360, True), use_container_width=True, theme=None)
    left, right = st.columns(2)
    with left:
        with st.container(border=True):
            card_title("Compared with the average phone", "100 = average of all phones")
            sp = ["ram", "battery_power", "int_memory", "px_width", "px_height"]
            rel = (df.groupby("tier")[sp].mean() / data[sp].mean() * 100).reindex(order).reset_index()
            rel = rel.melt(id_vars="tier", var_name="spec", value_name="index_value")
            rel["spec"] = rel["spec"].map(SPECS)
            fig = px.bar(rel, x="tier", y="index_value", color="spec", barmode="group",
                         labels={"tier": "Price tier", "index_value": "Index", "spec": "Specification"})
            fig.add_hline(y=100, line_dash="dash", line_color="grey")
            st.plotly_chart(style(fig, 360, True), use_container_width=True, theme=None)
    with right:
        with st.container(border=True):
            card_title("Features by price tier", "Share of phones that have each feature")
            fl = {"blue": "Bluetooth", "dual_sim": "Dual SIM", "four_g": "4G", "three_g": "3G",
                  "touch_screen": "Touch screen", "wifi": "Wi-Fi"}
            share = (df.groupby("tier")[list(fl)].mean() * 100).reindex(order).reset_index()
            share = share.melt(id_vars="tier", var_name="feature", value_name="share")
            share["feature"] = share["feature"].map(fl)
            fig = px.bar(share, x="feature", y="share", color="tier", barmode="group",
                         color_discrete_map=COLORS, category_orders={"tier": order},
                         labels={"feature": "Feature", "share": "Phones with the feature (%)", "tier": "Price tier"})
            st.plotly_chart(style(fig, 360, True), use_container_width=True, theme=None)

# ---------------- Phone explorer ----------------
elif page == "Phone explorer":
    header("Phone intelligence", "Phone explorer", "Choose a phone profile and see which price tiers those phones belong to.")
    df, order = filter_bar()
    RAM_GROUPS = {
        "Any RAM": (0, 10**6), "Under 1,000 MB": (0, 999), "1,000 to 1,999 MB": (1000, 1999),
        "2,000 to 2,999 MB": (2000, 2999), "3,000 MB and above": (3000, 10**6),
    }
    BAT_GROUPS = {
        "Any battery": (0, 10**6), "Under 800 mAh": (0, 799), "800 to 1,199 mAh": (800, 1199),
        "1,200 to 1,599 mAh": (1200, 1599), "1,600 mAh and above": (1600, 10**6),
    }
    with st.container(border=True):
        card_title("Phone profile", "Pick a RAM group and a battery group.")
        c1, c2 = st.columns(2)
        ram_choice = c1.selectbox("RAM", list(RAM_GROUPS), key="ram_choice")
        bat_choice = c2.selectbox("Battery", list(BAT_GROUPS), key="bat_choice")
    view = df[df["ram"].between(*RAM_GROUPS[ram_choice]) & df["battery_power"].between(*BAT_GROUPS[bat_choice])]
    k = st.columns(3)
    k[0].markdown(kpi("Matching phones", f"{len(view):,}", f"of {len(df):,} shown"), unsafe_allow_html=True)
    if view.empty:
        st.info("No phones match these choices. Try another group.")
    else:
        top = view["tier"].value_counts()
        k[1].markdown(kpi("Most common tier", top.index[0], f"{top.iloc[0] / len(view) * 100:.0f}% of matching phones"), unsafe_allow_html=True)
        k[2].markdown(kpi("Average RAM", f"{view['ram'].mean():,.0f} MB", "of matching phones"), unsafe_allow_html=True)
        st.write("")
        with st.container(border=True):
            card_title("Which price tiers do these phones belong to?", "Number of matching phones in each tier")
            counts = view.groupby("tier").size().reindex(order, fill_value=0).reset_index(name="phones")
            fig = px.bar(counts, x="tier", y="phones", color="tier", color_discrete_map=COLORS,
                         text="phones", labels={"tier": "Price tier", "phones": "Number of phones"})
            st.plotly_chart(style(fig, 340), use_container_width=True, theme=None)

# ---------------- Findings and insights ----------------
elif page == "Findings and insights":
    header("Insights", "Findings and insights", "What the analysis found, and what it means.")
    t = run_tests(data)
    models, results, cols, med = train_models(data)
    best = results.loc[results["Test accuracy (%)"].idxmax()]
    tm = data.groupby("price_range")[["ram", "battery_power", "int_memory"]].mean()
    p_model = stats.binomtest(int(best["correct"]), int(best["n"]), 0.25, alternative="greater").pvalue
    dec = lambda p: "Reject H0" if p < 0.05 else "Keep H0"

    with st.container(border=True):
        card_title("Research questions answered", "From the project proposal")
        st.markdown(
            f'<div class="tk"><b>RQ1. Which specifications matter most?</b><br><span>RAM (correlation {price_corr["ram"]:.2f}), then battery ({price_corr["battery_power"]:.2f}).</span></div>'
            f'<div class="tk"><b>RQ2. Does RAM differ across price ranges?</b><br><span>Yes. Average RAM rises from {tm["ram"][0]:,.0f} to {tm["ram"][3]:,.0f} MB (p {fp(t["p_ram"])}).</span></div>'
            f'<div class="tk"><b>RQ3. Does battery differ between low-cost and high-cost phones?</b><br><span>Yes, modestly: {tm["battery_power"][0]:,.0f} vs {tm["battery_power"][3]:,.0f} mAh (p {fp(t["p_t"])}).</span></div>'
            f'<div class="tk"><b>RQ4. How accurately can price range be predicted?</b><br><span>Very accurately: {best["Model"]} reaches {best["Test accuracy (%)"]}%.</span></div>',
            unsafe_allow_html=True)

    with st.container(border=True):
        card_title("Hypothesis tests", "Calculated live from the cleaned data")
        hc = st.columns(4)
        items = [("H1: RAM across tiers", dec(t["p_ram"]), f"F = {t['f_ram']:.1f}, p {fp(t['p_ram'])}"),
                 ("H2: Battery across tiers", dec(t["p_bat"]), f"F = {t['f_bat']:.1f}, p {fp(t['p_bat'])}"),
                 ("H3: Internal memory", dec(t["p_mem"]), f"rho = {t['rho']:.3f}, p {fp(t['p_mem'])}"),
                 ("H4: Model vs guessing", dec(p_model), f"{best['Test accuracy (%)']}%, p {fp(p_model)}")]
        for col, (name, d, note) in zip(hc, items):
            col.markdown(kpi(name, d, note), unsafe_allow_html=True)

    left, right = st.columns(2)
    with left:
        with st.container(border=True):
            card_title("Model comparison", "Test accuracy on phones the models had not seen")
            fig = px.bar(results, x="Model", y="Test accuracy (%)", text="Test accuracy (%)", color_discrete_sequence=["#0EA5E9"])
            fig.add_hline(y=25, line_dash="dash", line_color="grey", annotation_text="Random guessing (25%)")
            fig.update_yaxes(range=[0, 100])
            st.plotly_chart(style(fig, 340), use_container_width=True, theme=None)
    with right:
        with st.container(border=True):
            card_title("Typical RAM by price tier", "A phone with more RAM than its tier's typical amount offers more for that tier")
            mt = data.groupby("tier")["ram"].median().reindex(list(TIERS.values())).reset_index()
            fig = px.bar(mt, x="tier", y="ram", color="tier", color_discrete_map=COLORS, text="ram",
                         labels={"tier": "Price tier", "ram": "Median RAM (MB)"})
            st.plotly_chart(style(fig, 340), use_container_width=True, theme=None)

    with st.container(border=True):
        card_title("What it means", "Insights and limitations")
        step = (tm["ram"][3] - tm["ram"][0]) / 3
        st.markdown(
            f'<div class="tk"><b>RAM is the clearest sign of tier</b><br><span>The typical phone gains about {step:,.0f} MB of RAM with each step up.</span></div>'
            f'<div class="tk"><b>Battery and memory are weak signals</b><br><span>Internal memory only moves from {tm["int_memory"][0]:.0f} to {tm["int_memory"][3]:.0f} GB across tiers.</span></div>'
            f'<div class="tk"><b>For buyers</b><br><span>Compare RAM first when judging whether a phone matches its price tier.</span></div>'
            f'<div class="tk"><b>Limitations</b><br><span>The data has price tiers, not prices. Classes are perfectly balanced, so the data may not mirror the real market. Models were not tuned.</span></div>',
            unsafe_allow_html=True)

# ---------------- Price predictor ----------------
elif page == "Price predictor":
    header("Prediction", "Price predictor", "Enter a phone's specifications to predict its price tier.")
    models, results, cols, med = train_models(data)
    lr = models["Logistic Regression"]
    acc = results.loc[results["Model"] == "Logistic Regression", "Test accuracy (%)"].iloc[0]

    def num(c, label, step, box):
        return box.number_input(label, int(data[c].min()), int(data[c].max()), int(med[c]), step=step)

    with st.container(border=True):
        card_title("Phone specifications", "Anything not entered here uses the typical (median) value from the dataset.")
        c1, c2, c3 = st.columns(3)
        row = med.copy()
        row["ram"] = num("ram", "RAM (MB)", 50, c1)
        row["battery_power"] = num("battery_power", "Battery (mAh)", 50, c2)
        row["int_memory"] = num("int_memory", "Internal memory (GB)", 2, c3)
        c4, c5, c6 = st.columns(3)
        row["px_width"] = num("px_width", "Screen pixel width", 50, c4)
        row["px_height"] = num("px_height", "Screen pixel height", 50, c5)
        row["mobile_wt"] = num("mobile_wt", "Weight (g)", 5, c6)
        b = st.columns(6)
        for box, (c, label) in zip(b, [("blue", "Bluetooth"), ("dual_sim", "Dual SIM"), ("four_g", "4G"),
                                       ("three_g", "3G"), ("touch_screen", "Touch screen"), ("wifi", "Wi-Fi")]):
            row[c] = int(box.checkbox(label, value=bool(med[c]), key="chk_" + c))

    X_new = pd.DataFrame([row], columns=cols)
    pred = int(lr.predict(X_new)[0])
    proba = lr.predict_proba(X_new)[0]
    left, right = st.columns([1, 2])
    with left:
        st.markdown(kpi("Predicted price tier", TIERS[pred], f"confidence {proba.max() * 100:.0f}%"), unsafe_allow_html=True)
        st.markdown(f'<div class="cs">Logistic Regression, {acc}% accuracy on test phones.</div>', unsafe_allow_html=True)
    with right:
        with st.container(border=True):
            card_title("Probability of each tier", "How sure the model is")
            fig = px.bar(x=list(TIERS.values()), y=proba * 100, color=list(TIERS.values()),
                         color_discrete_map=COLORS, labels={"x": "Price tier", "y": "Probability (%)"})
            st.plotly_chart(style(fig, 280), use_container_width=True, theme=None)

# ---------------- Reports ----------------
elif page == "Reports":
    header("Export center", "Reports", "Download your cleaned data and summaries.")
    df, order = filter_bar()
    summary = df.groupby("tier")[list(SPECS)].mean().round(1).reset_index()
    r1, r2, r3 = st.columns(3)
    with r1:
        with st.container(border=True):
            card_title("Cleaned dataset", f"{len(df):,} phones in the selected tiers")
            st.download_button("Download CSV", df.to_csv(index=False), "clean_phones.csv", key="d1")
    with r2:
        with st.container(border=True):
            card_title("Tier summary", "Average specifications by price tier")
            st.download_button("Download CSV", summary.to_csv(index=False), "tier_summary.csv", key="d2")
    with r3:
        with st.container(border=True):
            card_title("Correlation table", "Correlation of every specification")
            st.download_button("Download CSV", corr.round(3).to_csv(), "correlations.csv", key="d3")

# ---------------- User guide ----------------
else:
    header("Help", "User guide", "How to find your way around the dashboard.")
    with st.container(border=True):
        card_title("What each page does", "Use the menu on the left")
        st.markdown("""
- **Overview:** key numbers, average and spread of a specification by price tier, and feature ranking.
- **Specification trends:** see how a specification is spread across the tiers, how each tier compares with the average phone, and which features each tier has.
- **Phone explorer:** choose a RAM group and a battery group to see which price tiers those phones belong to.
- **Findings and insights:** research questions, hypothesis tests, model comparison and conclusions, shown as cards and charts.
- **Price predictor:** type in a phone's specifications to predict its price tier.
- **Reports:** download the cleaned data and summary tables as CSV files.
""")
    with st.container(border=True):
        card_title("Filters", "Price tier tick boxes at the top of the page")
        st.markdown("Use the tick boxes at the top of the Overview, Specification trends, Phone explorer and Reports pages to focus on some price tiers, and click **Reset filters** to bring all four back. You can also click a tier name in a chart's legend to hide or show it. Findings and the predictor always use all phones.")
    with st.container(border=True):
        card_title("Data preparation", "How the data was cleaned")
        st.markdown("""
- Source: Mobile Price Classification dataset (Kaggle), 2,000 phones, 20 specifications and a price range (0 to 3).
- No missing values or duplicates were found.
- 2 phones with a screen pixel height of 0 were removed. 180 phones with a screen width of 0 were filled with the median width.
- Outliers (front camera, pixel height) were checked with the IQR method and kept because they are realistic.
- Result: 1,998 phones.
""")
