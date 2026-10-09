import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

st.set_page_config(page_title="AdPulse | Sales Lab", page_icon="📊", layout="wide")

# A custom visual identity: midnight navy, teal, and coral.
st.markdown("""
<style>
:root { --ink:#172554; --teal:#0f766e; --coral:#fb7185; }
.stApp { background: linear-gradient(135deg, #f0fdfa 0%, #f8fafc 52%, #fff1f2 100%); }
[data-testid="stSidebar"] { background: #172554; }
[data-testid="stSidebar"] * { color: #f8fafc !important; }
.hero { padding: 1.5rem 1.8rem; border-radius: 22px; background: linear-gradient(110deg,#172554,#0f766e); color: white; margin-bottom: 1rem; }
.hero h1 { color:white; margin:0; font-size:2.15rem; }
.hero p { color:#ccfbf1; margin:.5rem 0 0 0; }
.metric-card { background:white; border:1px solid #ccfbf1; border-left:5px solid #0f766e; border-radius:15px; padding:1rem; box-shadow:0 5px 20px #0f172a0b; }
.metric-label { color:#64748b; font-size:.88rem; }
.metric-value { color:#172554; font-size:1.6rem; font-weight:750; }
div[data-testid="stTabs"] button { font-weight:650; }
</style>
<div class="hero"><h1>AdPulse · Sales Analytics Lab</h1><p>Advertising spend → predicted sales | Built by Ashok</p></div>
""", unsafe_allow_html=True)

@st.cache_data
def load_data():
    return pd.read_csv("Advertising.csv")

df = load_data()
required = {"TV", "Radio", "Newspaper", "Sales"}
if not required.issubset(df.columns):
    st.error("Advertising.csv must contain TV, Radio, Newspaper, and Sales columns.")
    st.stop()

features = ["TV", "Radio", "Newspaper"]
X, y = df[features], df["Sales"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)
model = LinearRegression().fit(X_train, y_train)
predictions = model.predict(X_test)
mae = mean_absolute_error(y_test, predictions)
mse = mean_squared_error(y_test, predictions)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, predictions)

with st.sidebar:
    st.title("ADPULSE")
    st.caption("MULTIPLE LINEAR REGRESSION")
    st.markdown("---")
    st.markdown("### Navigation")
    page = st.radio("Choose a workspace", ["Overview", "Sales Predictor", "Model Insights", "Dataset Explorer"], label_visibility="collapsed")
    st.markdown("---")
    st.caption("Model setup")
    st.write(f"Training rows: **{len(X_train)}**")
    st.write(f"Test rows: **{len(X_test)}**")
    st.write("Split: **80 / 20**")
    st.write("Random state: **42**")

if page == "Overview":
    st.subheader("Performance at a glance")
    cols = st.columns(4)
    items = [("DATA RECORDS", f"{len(df):,}"), ("INPUT CHANNELS", "3"), ("R² SCORE", f"{r2:.3f}"), ("RMSE", f"{rmse:.3f}")]
    for col, (label, value) in zip(cols, items):
        with col:
            st.markdown(f'<div class="metric-card"><div class="metric-label">{label}</div><div class="metric-value">{value}</div></div>', unsafe_allow_html=True)
    st.write("")
    left, right = st.columns([1.15, 1])
    with left:
        st.markdown("#### Advertising mix in the dataset")
        fig, ax = plt.subplots(figsize=(7, 4))
        totals = df[features].sum().sort_values(ascending=False)
        ax.bar(totals.index, totals.values, color=["#0f766e", "#fb7185", "#6366f1"])
        ax.set_ylabel("Combined spend (dataset units)")
        ax.set_facecolor("#ffffff")
        fig.patch.set_facecolor("#ffffff")
        ax.spines[["top", "right"]].set_visible(False)
        st.pyplot(fig, use_container_width=True)
    with right:
        st.markdown("#### Project workflow")
        st.markdown("""
        1. Load and inspect advertising records.
        2. Choose TV, Radio, and Newspaper as predictors.
        3. Split records into training and test sets.
        4. Fit a Multiple Linear Regression model.
        5. Predict sales and review error metrics.
        6. Explore coefficients and visual comparisons.
        """)
        st.info("Use the Sales Predictor tab to test a custom advertising budget.")
elif page == "Sales Predictor":
    st.subheader("Estimate sales from a planned budget")
    st.write("Adjust the three channels. The trained model estimates the corresponding sales value.")
    a, b = st.columns([1, 1.2])
    with a:
        tv = st.slider("TV advertising", 0.0, float(max(300, df.TV.max())), 150.0, 1.0)
        radio = st.slider("Radio advertising", 0.0, float(max(60, df.Radio.max())), 20.0, 0.5)
        newspaper = st.slider("Newspaper advertising", 0.0, float(max(120, df.Newspaper.max())), 30.0, 0.5)
        input_df = pd.DataFrame({"TV": [tv], "Radio": [radio], "Newspaper": [newspaper]})
        result = float(model.predict(input_df)[0])
        st.markdown(f'<div class="metric-card"><div class="metric-label">ESTIMATED SALES</div><div class="metric-value">{result:.2f}</div><div class="metric-label">Sales units as represented by the dataset</div></div>', unsafe_allow_html=True)
    with b:
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.bar(["TV", "Radio", "Newspaper"], [tv, radio, newspaper], color=["#0f766e", "#fb7185", "#6366f1"])
        ax.set_ylabel("Advertising input")
        ax.set_title("Your budget mix")
        ax.spines[["top", "right"]].set_visible(False)
        fig.patch.set_facecolor("white")
        st.pyplot(fig, use_container_width=True)
        st.caption("Prediction is an estimate from a linear model, not a guarantee of actual sales.")
elif page == "Model Insights":
    st.subheader("How well does the model perform?")
    m1, m2, m3, m4 = st.columns(4)
    for col, label, value in [(m1,"MAE",mae),(m2,"MSE",mse),(m3,"RMSE",rmse),(m4,"R²",r2)]:
        with col: st.metric(label, f"{value:.4f}")
    st.markdown("#### Actual versus predicted sales")
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.scatter(y_test, predictions, color="#0f766e", edgecolor="white", s=65, alpha=.85)
    lo, hi = min(y_test.min(), predictions.min()), max(y_test.max(), predictions.max())
    ax.plot([lo, hi], [lo, hi], linestyle="--", color="#fb7185", linewidth=2, label="Ideal prediction")
    ax.set_xlabel("Actual sales")
    ax.set_ylabel("Predicted sales")
    ax.legend(frameon=False)
    ax.spines[["top", "right"]].set_visible(False)
    fig.patch.set_facecolor("white")
    st.pyplot(fig, use_container_width=True)
    st.markdown("#### Learned coefficients")
    coef_df = pd.DataFrame({"Advertising channel": features, "Coefficient": model.coef_}).sort_values("Coefficient", ascending=False)
    st.dataframe(coef_df, use_container_width=True, hide_index=True)
    st.caption(f"Intercept: {model.intercept_:.4f}. Coefficients describe estimated change in sales for one unit of a channel, holding the other channels fixed.")
elif page == "Dataset Explorer":
    st.subheader("Inspect the source records")
    c1, c2, c3 = st.columns(3)
    c1.metric("Rows", len(df)); c2.metric("Columns", len(df.columns)); c3.metric("Missing cells", int(df.isna().sum().sum()))
    st.dataframe(df, use_container_width=True, height=350)
    st.markdown("#### Descriptive statistics")
    st.dataframe(df.describe().T, use_container_width=True)
    st.download_button("Download dataset CSV", df.to_csv(index=False).encode("utf-8"), "Advertising.csv", "text/csv")

st.markdown("---")
st.caption("AdPulse · Sales Analytics Lab | Prepared by Ashok | Educational analysis")
