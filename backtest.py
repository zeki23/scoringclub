# app.py
import io
import math
import numpy as np
import pandas as pd
import yfinance as yf
import streamlit as st
from datetime import datetime

st.set_page_config(page_title="Backtest simple", layout="wide")

# ========= Helpers =========
@st.cache_data
def fetch_prices(tickers, start, end):
    data = yf.download(tickers, start=start, end=end, auto_adjust=True, progress=False)
    if isinstance(data.columns, pd.MultiIndex):
        close = data["Close"]
    else:
        close = data.to_frame(name="Close")
        close.columns = [tickers[0]]
    return close

def perf_stats(returns: pd.Series, periods_per_year=252):
    r = returns.dropna()
    if len(r) == 0:
        return dict(CAGR=np.nan, Vol=np.nan, Sharpe=np.nan, MaxDD=np.nan, Cum=np.nan)
    equity = (1 + r).cumprod()
    cum = equity.iloc[-1] - 1
    years = len(r) / periods_per_year
    cagr = equity.iloc[-1] ** (1/years) - 1 if years > 0 else np.nan
    vol = r.std() * math.sqrt(periods_per_year)
    sharpe = (r.mean() * periods_per_year) / vol if vol > 0 else np.nan
    mdd = (equity / equity.cummax() - 1).min()
    return dict(CAGR=cagr, Vol=vol, Sharpe=sharpe, MaxDD=mdd, Cum=cum)

def drawdown_series(returns: pd.Series) -> pd.Series:
    eq = (1 + returns).cumprod()
    return eq / eq.cummax() - 1

def is_oui_non_column(s: pd.Series) -> bool:
    vals = s.dropna().astype(str).str.upper()
    return (vals.isin(["OUI", "NON"]).mean() > 0.5)

# ========= Sidebar =========
st.sidebar.header("Paramètres Backtest")
mode = st.sidebar.radio("Source des tickers", ["Charger un Excel", "Saisir manuellement"])
start = st.sidebar.date_input("Début", value=datetime(2020, 1, 1))
end = st.sidebar.date_input("Fin", value=datetime.today())
bench = st.sidebar.text_input("Benchmark (Yahoo)", value="SPY")

st.title("Backtest simple")

# ========= Sélection des tickers =========
tickers = []

if mode == "Charger un Excel":
    up = st.file_uploader("Fichier Excel (doit contenir une colonne 'Ticker')", type=["xlsx"])
    if up is None:
        st.info("Charge un fichier Excel pour continuer.")
        st.stop()

    try:
        df = pd.read_excel(up)
    except Exception as e:
        st.error(f"Lecture Excel impossible : {e}")
        st.stop()

    if "Ticker" not in df.columns:
        st.error("Le fichier doit contenir une colonne 'Ticker'.")
        st.stop()

    st.subheader("Aperçu du fichier")
    st.dataframe(df.head(20), use_container_width=True)

    # Détecte des colonnes de filtres OUI/NON
    filter_cols = [c for c in df.columns if ("Filtre" in c) or is_oui_non_column(df[c])]
    filter_cols = [c for c in filter_cols if c != "Ticker"]
    filter_cols = list(dict.fromkeys(filter_cols))

    if len(filter_cols) > 0:
        st.subheader("Filtres (OUI requis)")
        selected_filters = st.multiselect(
            "Choisis les filtres à appliquer (les lignes doivent être 'OUI' pour chacun) :",
            options=filter_cols,
            default=[]
        )
        df_use = df.copy()
        for f in selected_filters:
            df_use = df_use[df_use[f].astype(str).str.upper() == "OUI"]
    else:
        df_use = df

    tickers = sorted(df_use["Ticker"].dropna().astype(str).unique().tolist())

else:
    tickers_text = st.text_area(
        "Entre des tickers séparés par virgules ou retours à la ligne (ex: AAPL, MSFT, NVDA)",
        height=120,
    )
    if not tickers_text.strip():
        st.info("Entre au moins un ticker pour continuer.")
        st.stop()
    raw = [t.strip().upper() for t in tickers_text.replace("\n", ",").split(",")]
    tickers = sorted([t for t in raw if t])

if len(tickers) == 0:
    st.error("Aucun ticker sélectionné.")
    st.stop()

st.subheader("Tickers retenus")
st.write(f"{len(tickers)} tickers :", ", ".join(tickers[:30]) + (" ..." if len(tickers) > 30 else ""))

# ========= Téléchargement des prix =========
with st.spinner("Téléchargement des prix…"):
    px = fetch_prices(tickers, start=start, end=end)
    bench_px = fetch_prices([bench], start=start, end=end)

if px.empty or bench_px.empty:
    st.error("Données de marché introuvables pour la période choisie.")
    st.stop()

# Aligne les dates
data = px.join(bench_px, how="inner", rsuffix="_bench").dropna(how="all")
if data.empty:
    st.error("Pas de recouvrement de dates entre tickers et benchmark.")
    st.stop()

# Sépare
bench_series = data.iloc[:, -1].rename(bench)
prices = data.iloc[:, :-1]

# ========= Rendements & Portefeuille =========
rets = prices.pct_change().dropna(how="all").fillna(0)
bench_rets = bench_series.pct_change().dropna().fillna(0)

# Portefeuille égal-pondéré (moyenne simple des rendements quotidiens)
port_rets = rets.mean(axis=1).dropna()

# ========= Stats =========
stats_port = perf_stats(port_rets)
stats_bench = perf_stats(bench_rets)

col1, col2 = st.columns(2)
with col1:
    st.markdown("### 📈 Portefeuille")
    st.metric("CAGR", f"{stats_port['CAGR']*100:,.2f}%")
    st.metric("Vol annualisée", f"{stats_port['Vol']*100:,.2f}%")
    st.metric("Sharpe (≈)", f"{stats_port['Sharpe']:.2f}")
    st.metric("Max Drawdown", f"{stats_port['MaxDD']*100:,.2f}%")
    st.metric("Perf cumulée", f"{stats_port['Cum']*100:,.2f}%")

with col2:
    st.markdown(f"### 🧭 Benchmark ({bench})")
    st.metric("CAGR", f"{stats_bench['CAGR']*100:,.2f}%")
    st.metric("Vol annualisée", f"{stats_bench['Vol']*100:,.2f}%")
    st.metric("Sharpe (≈)", f"{stats_bench['Sharpe']:.2f}")
    st.metric("Max Drawdown", f"{stats_bench['MaxDD']*100:,.2f}%")
    st.metric("Perf cumulée", f"{stats_bench['Cum']*100:,.2f}%")

# ========= Graphiques =========
st.markdown("### Courbe d’équité")
eq_df = pd.DataFrame({
    "Portefeuille": (1 + port_rets).cumprod(),
    bench: (1 + bench_rets).cumprod()
}).dropna()
st.line_chart(eq_df, use_container_width=True)

st.markdown("### Drawdown")
dd_df = pd.DataFrame({
    "Portefeuille": drawdown_series(port_rets),
    bench: drawdown_series(bench_rets)
}).dropna()
st.area_chart(dd_df, use_container_width=True)

st.markdown("### Rendements quotidiens")
st.bar_chart(port_rets, use_container_width=True)

# ========= Export =========
st.markdown("### Export")
out = io.BytesIO()
with pd.ExcelWriter(out, engine="openpyxl") as writer:
    eq_df.to_excel(writer, sheet_name="equity_curve")
    dd_df.to_excel(writer, sheet_name="drawdown")
    pd.DataFrame(port_rets, columns=["port_returns"]).to_excel(writer, sheet_name="port_returns")
    pd.DataFrame(bench_rets, columns=[f"{bench}_returns"]).to_excel(writer, sheet_name="bench_returns")
    pd.DataFrame([stats_port]).to_excel(writer, sheet_name="port_stats", index=False)
    pd.DataFrame([stats_bench]).to_excel(writer, sheet_name="bench_stats", index=False)

st.download_button(
    label="📥 Télécharger les résultats (Excel)",
    data=out.getvalue(),
    file_name="backtest_simple_results.xlsx",
    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
)
