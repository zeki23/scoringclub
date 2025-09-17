# app.py
import io
import time
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import streamlit as st
import yfinance as yf

st.set_page_config(page_title="Screener fondamental (yfinance) → Excel", layout="wide")

# =============================
# Helpers UI
# =============================
st.title("Screener fondamental → Export Excel")
st.caption("Entrez des tickers (ex: MSFT, AAPL, RI.PA) ou importez un CSV avec une colonne 'Ticker' (+ 'Secteur' optionnelle).")

with st.sidebar:
    st.header("Paramètres Scoring")
    sleep_sec = st.number_input("Pause (sec) entre tickers", min_value=0.0, max_value=10.0, value=0.0, step=0.5)
    debug = st.toggle("Mode DEBUG", value=False)


# =============================
# Fonctions utilitaires
# =============================
def find_row_label(df: pd.DataFrame, target_keywords: List[str]) -> Optional[str]:
    for row in df.index:
        if isinstance(row, str):
            row_lower = row.lower()
            for key in target_keywords:
                if key.lower() in row_lower:
                    return row
    return None

def safe_getattr(obj, name, default=None):
    try:
        val = getattr(obj, name, default)
        return val if val is not None else default
    except Exception:
        return default

# =============================
# Analyse d’un ticker
# =============================
def analyze_ticker(ticker: str, secteurs_hint: Dict[str, str], debug: bool = False) -> Dict:
    data = {"Ticker": ticker}
    try:
        stock = yf.Ticker(ticker)
        info = safe_getattr(stock, "info", {}) or {}

        # Secteur
        data["Secteur"] = secteurs_hint.get(ticker) or info.get("sector")

        # États financiers (compat yfinance)
        # Certaines versions exposent .financials/.income_stmt/.cashflow ; on gère les deux formats
        financials = safe_getattr(stock, "financials", pd.DataFrame())
        income_stmt = safe_getattr(stock, "income_stmt", pd.DataFrame())
        cashflow_stmt = safe_getattr(stock, "cashflow", pd.DataFrame())

        # Uniformiser l'orientation (on veut des lignes = périodes, colonnes = items)
        def orient(df):
            if isinstance(df, pd.DataFrame) and not df.empty:
                # yfinance renvoie souvent index=Items, columns=dates -> transpose
                if df.index.size > df.columns.size:
                    return df.T.sort_index()
                else:
                    return df.sort_index()
            return pd.DataFrame()

        financials_T = orient(financials)
        income_T = orient(income_stmt)
        cashflow_T = orient(cashflow_stmt)

        # --- Revenue Growth (N vs N-4)
        if financials_T.shape[0] >= 4 and "Total Revenue" in financials_T.columns:
            rev_now = financials_T.iloc[-1]["Total Revenue"]
            rev_past = financials_T.iloc[-4]["Total Revenue"]
            if pd.notnull(rev_now) and pd.notnull(rev_past):
                growth = (rev_now - rev_past) / abs(rev_past) * 100
                data["Revenue N"] = round(rev_now / 1e9, 2)
                data["Revenue N-4"] = round(rev_past / 1e9, 2)
                data["Revenue Growth (%)"] = round(growth, 2)
                data["Filtre Revenue Growth"] = "OUI" if rev_now > rev_past else "NON"
            else:
                data["Revenue N"] = data["Revenue N-4"] = data["Revenue Growth (%)"] = None
                data["Filtre Revenue Growth"] = "NON"
        else:
            data["Revenue N"] = data["Revenue N-4"] = data["Revenue Growth (%)"] = None
            data["Filtre Revenue Growth"] = "NON"

        # --- EBITDA Growth
        if income_T.shape[0] >= 4 and "EBITDA" in income_T.columns:
            ebitda_now = income_T.iloc[-1]["EBITDA"]
            ebitda_4y = income_T.iloc[-4]["EBITDA"]
            if pd.notnull(ebitda_now) and pd.notnull(ebitda_4y):
                growth_ebitda = (ebitda_now - ebitda_4y) / abs(ebitda_4y) * 100
                data["EBITDA N"] = round(ebitda_now / 1e9, 2)
                data["EBITDA N-4"] = round(ebitda_4y / 1e9, 2)
                data["EBITDA Growth (%)"] = round(growth_ebitda, 2)
                data["Filtre EBITDA Growth"] = "OUI" if ebitda_now > ebitda_4y else "NON"
            else:
                data["EBITDA N"] = data["EBITDA N-4"] = data["EBITDA Growth (%)"] = None
                data["Filtre EBITDA Growth"] = "NON"
        else:
            data["EBITDA N"] = data["EBITDA N-4"] = data["EBITDA Growth (%)"] = None
            data["Filtre EBITDA Growth"] = "NON"

        # --- Net Income Growth
        if income_T.shape[0] >= 4 and "Net Income" in income_T.columns:
            net_now = income_T.iloc[-1]["Net Income"]
            net_4y = income_T.iloc[-4]["Net Income"]
            if pd.notnull(net_now) and pd.notnull(net_4y):
                growth_net = (net_now - net_4y) / abs(net_4y) * 100
                data["Net Income N"] = round(net_now / 1e9, 2)
                data["Net Income N-4"] = round(net_4y / 1e9, 2)
                data["Net Income Growth (%)"] = round(growth_net, 2)
                data["Filtre Net Income Growth"] = "OUI" if net_now > net_4y else "NON"
            else:
                data["Net Income N"] = data["Net Income N-4"] = data["Net Income Growth (%)"] = None
                data["Filtre Net Income Growth"] = "NON"
        else:
            data["Net Income N"] = data["Net Income N-4"] = data["Net Income Growth (%)"] = None
            data["Filtre Net Income Growth"] = "NON"

        # --- EPS Growth
        if income_T.shape[0] >= 4 and "Diluted EPS" in income_T.columns:
            eps_now = income_T.iloc[-1]["Diluted EPS"]
            eps_4y = income_T.iloc[-4]["Diluted EPS"]
            if pd.notnull(eps_now) and pd.notnull(eps_4y):
                growth_eps = (eps_now - eps_4y) / abs(eps_4y) * 100
                data["EPS N"] = round(float(eps_now), 2)
                data["EPS N-4"] = round(float(eps_4y), 2)
                data["EPS Growth (%)"] = round(growth_eps, 2)
                data["Filtre EPS Growth"] = "OUI" if eps_now > eps_4y else "NON"
            else:
                data["EPS N"] = data["EPS N-4"] = data["EPS Growth (%)"] = None
                data["Filtre EPS Growth"] = "NON"
        else:
            data["EPS N"] = data["EPS N-4"] = data["EPS Growth (%)"] = None
            data["Filtre EPS Growth"] = "NON"

        # --- FCFE Growth = OCF + CAPEX (capex < 0)
        if cashflow_T.shape[0] >= 4:
            op_cols = ["Operating Cash Flow", "Total Cash From Operating Activities"]
            capex_cols = ["Capital Expenditure", "Capital Expenditures"]

            op_cf_now = next((cashflow_T.iloc[-1].get(c) for c in op_cols if c in cashflow_T.columns), None)
            op_cf_4y = next((cashflow_T.iloc[-4].get(c) for c in op_cols if c in cashflow_T.columns), None)
            capex_now = next((cashflow_T.iloc[-1].get(c) for c in capex_cols if c in cashflow_T.columns), None)
            capex_4y = next((cashflow_T.iloc[-4].get(c) for c in capex_cols if c in cashflow_T.columns), None)

            fcfe_now = (op_cf_now + capex_now) if pd.notnull(op_cf_now) and pd.notnull(capex_now) else None
            fcfe_4y = (op_cf_4y + capex_4y) if pd.notnull(op_cf_4y) and pd.notnull(capex_4y) else None

            if pd.notnull(fcfe_now) and pd.notnull(fcfe_4y) and fcfe_4y != 0:
                growth_fcfe = (fcfe_now - fcfe_4y) / abs(fcfe_4y) * 100
                data["FCFE N"] = round(fcfe_now / 1e9, 2)
                data["FCFE N-4"] = round(fcfe_4y / 1e9, 2)
                data["FCFE Growth (%)"] = round(growth_fcfe, 2)
                data["Filtre FCFE Growth"] = "OUI" if fcfe_now > fcfe_4y else "NON"
            else:
                data["FCFE N"] = data["FCFE N-4"] = data["FCFE Growth (%)"] = None
                data["Filtre FCFE Growth"] = "NON"
        else:
            data["FCFE N"] = data["FCFE N-4"] = data["FCFE Growth (%)"] = None
            data["Filtre FCFE Growth"] = "NON"

        # --- ROE (approché)
        try:
            latest_year = income_T.index[-1] if not income_T.empty else None
            bs = safe_getattr(stock, "balance_sheet", pd.DataFrame())
            if not bs.empty and latest_year is not None:
                # balance_sheet typiquement: index=items, columns=dates
                common_stock = bs.at["Common Stock", latest_year] if "Common Stock" in bs.index and latest_year in bs.columns else 0
                apic = bs.at["Additional Paid In Capital", latest_year] if "Additional Paid In Capital" in bs.index and latest_year in bs.columns else 0
                retained = bs.at["Retained Earnings", latest_year] if "Retained Earnings" in bs.index and latest_year in bs.columns else 0
                aoci = bs.at["Accumulated Other Comprehensive Income", latest_year] if "Accumulated Other Comprehensive Income" in bs.index and latest_year in bs.columns else 0
                common_equity = (common_stock or 0) + (apic or 0) + (retained or 0) + (aoci or 0)

                net_income_label = find_row_label(income_stmt, ["Net Income"])
                net_income = income_stmt.at[net_income_label, latest_year] if net_income_label and latest_year in income_stmt.columns else None

                if pd.notnull(net_income) and common_equity not in (None, 0):
                    roe = net_income / common_equity
                    data["ROE (%)"] = round(float(roe) * 100, 2)
                    data["Filtre ROE ≥ 15%"] = "OUI" if (roe * 100) >= 15 else "NON"
                else:
                    data["ROE (%)"] = None
                    data["Filtre ROE ≥ 15%"] = "NON"
            else:
                data["ROE (%)"] = None
                data["Filtre ROE ≥ 15%"] = "NON"
        except Exception as e:
            if debug: st.write(f"[DEBUG] ROE error {ticker}: {e}")
            data["ROE (%)"] = None
            data["Filtre ROE ≥ 15%"] = "NON"

        # --- Current Ratio (2024 si dispo)
        try:
            bs = safe_getattr(stock, "balance_sheet", pd.DataFrame())
            date_2024 = None
            if not bs.empty:
                for col in bs.columns:
                    if "2024" in str(col):
                        date_2024 = col
                        break
            if date_2024:
                current_assets = bs.loc["Current Assets", date_2024]
                current_liabilities = bs.loc["Current Liabilities", date_2024]
                curr_ratio = (current_assets / current_liabilities) if current_liabilities else None
                data["Current Assets (bn)"] = round(current_assets / 1e9, 2) if pd.notnull(current_assets) else None
                data["Current Liabilities (bn)"] = round(current_liabilities / 1e9, 2) if pd.notnull(current_liabilities) else None
                data["Current Ratio"] = round(curr_ratio, 2) if curr_ratio is not None else None
                data["Filtre Current Ratio ≥ 1.5"] = "OUI" if curr_ratio is not None and curr_ratio >= 1.5 else "NON"
            else:
                data["Current Ratio"] = None
                data["Filtre Current Ratio ≥ 1.5"] = "NON"
        except Exception as e:
            if debug: st.write(f"[DEBUG] Current Ratio error {ticker}: {e}")
            data["Current Ratio"] = None
            data["Filtre Current Ratio ≥ 1.5"] = "NON"

        # --- Debt-to-Equity ≤ 100
        dte = info.get("debtToEquity")
        data["Debt-to-Equity"] = round(float(dte), 2) if pd.notnull(dte) else None
        data["Filtre Debt-to-Equity ≤ 100"] = "OUI" if (pd.notnull(dte) and dte <= 100) else "NON"

        # --- Net & Operating Margin
        try:
            fin = safe_getattr(stock, "financials", pd.DataFrame())
            if fin is None or fin.empty or fin.columns.empty:
                data["Net Margin (%)"] = None
                data["Operating Margin (%)"] = None
                data["Filtre Net Margin ≥ 25%"] = "NON"
            else:
                latest_col = fin.columns[0]
                net_income_label = find_row_label(fin, ["Net Income"])
                op_income_label = find_row_label(fin, ["Operating Income"])
                revenue_label = find_row_label(fin, ["Total Revenue"])

                net_income = fin.at[net_income_label, latest_col] if net_income_label else None
                op_income = fin.at[op_income_label, latest_col] if op_income_label else None
                revenue = fin.at[revenue_label, latest_col] if revenue_label else None

                if pd.notnull(net_income) and pd.notnull(revenue) and revenue != 0:
                    nm = (net_income / revenue) * 100
                    data["Net Margin (%)"] = round(float(nm), 2)
                    data["Filtre Net Margin ≥ 25%"] = "OUI" if nm >= 25 else "NON"
                else:
                    data["Net Margin (%)"] = None
                    data["Filtre Net Margin ≥ 25%"] = "NON"

                if pd.notnull(op_income) and pd.notnull(revenue) and revenue != 0:
                    om = (op_income / revenue) * 100
                    data["Operating Margin (%)"] = round(float(om), 2)
                else:
                    data["Operating Margin (%)"] = None
        except Exception as e:
            if debug: st.write(f"[DEBUG] Margin error {ticker}: {e}")
            data["Net Margin (%)"] = None
            data["Operating Margin (%)"] = None
            data["Filtre Net Margin ≥ 25%"] = "NON"

        # --- FCF Margin ≥ 10%
        try:
            cf = cashflow_stmt
            fin = safe_getattr(stock, "financials", pd.DataFrame())
            latest_col_cf = cf.columns[0] if (isinstance(cf, pd.DataFrame) and not cf.empty) else None

            ocf_label = "Operating Cash Flow"
            capex_label = "Capital Expenditure"
            revenue_label = "Total Revenue"

            ocf = cf.at[ocf_label, latest_col_cf] if latest_col_cf in (cf.columns if isinstance(cf, pd.DataFrame) else []) and ocf_label in cf.index else None
            capex = cf.at[capex_label, latest_col_cf] if latest_col_cf in (cf.columns if isinstance(cf, pd.DataFrame) else []) and capex_label in cf.index else None

            # revenue au même dernier exercice des financials
            latest_col_fin = fin.columns[0] if (isinstance(fin, pd.DataFrame) and not fin.empty) else None
            revenue = fin.at[revenue_label, latest_col_fin] if latest_col_fin and revenue_label in fin.index else None

            if all(pd.notnull(x) for x in [ocf, capex, revenue]) and revenue != 0:
                fcf = ocf + capex
                fcf_margin = (fcf / revenue) * 100
                data["FCF Margin (%)"] = round(float(fcf_margin), 2)
                data["Filtre FCF Margin ≥ 10%"] = "OUI" if fcf_margin >= 10 else "NON"
            else:
                data["FCF Margin (%)"] = None
                data["Filtre FCF Margin ≥ 10%"] = "NON"
        except Exception as e:
            if debug: st.write(f"[DEBUG] FCF Margin error {ticker}: {e}")
            data["FCF Margin (%)"] = None
            data["Filtre FCF Margin ≥ 10%"] = "NON"

        # --- P/E & Forward P/E
        trailing_pe = info.get("trailingPE")
        forward_pe = info.get("forwardPE")
        data["Trailing P/E"] = round(float(trailing_pe), 2) if pd.notnull(trailing_pe) else None
        data["Forward P/E"] = round(float(forward_pe), 2) if pd.notnull(forward_pe) else None
        data["Filtre Forward P/E < Trailing P/E"] = (
            "OUI" if (pd.notnull(trailing_pe) and pd.notnull(forward_pe) and forward_pe < trailing_pe) else "NON"
        )

        

    except Exception as e:
        if debug:
            st.warning(f"Erreur pour {ticker}: {e}")

        # Si aucune donnée trouvée
    # ✅ À coller tout en bas de analyze_ticker, juste avant "return data"
    metrics_keys = [
        "Revenue N","EBITDA N","Net Income N","EPS N","FCFE N",
        "ROE (%)","Current Ratio","Debt-to-Equity",
        "Net Margin (%)","Operating Margin (%)","FCF Margin (%)",
        "Trailing P/E","Forward P/E",
    ]
    if not any(data.get(k) is not None for k in metrics_keys):
        st.error(f"Aucune donnée trouvée pour le ticker {ticker} : soit le ticker est faux, soit l'entreprise n'est pas cotée.")



    return data

# =============================
# Entrées utilisateur
# =============================
col1, col2 = st.columns(2)
with col1:
    tickers_text = st.text_area("Tickers (séparés par des virgules)", placeholder="MSFT, AAPL, RI.PA").strip()
with col2:
    csv_file = st.file_uploader("…ou importe un CSV", type=["csv"])

secteurs_hint: Dict[str, str] = {}
tickers: List[str] = []

if csv_file is not None:
    df_csv = pd.read_csv(csv_file)
    if "Ticker" not in df_csv.columns:
        st.error("Le CSV doit contenir une colonne 'Ticker'.")
        st.stop()
    tickers = df_csv["Ticker"].dropna().astype(str).unique().tolist()
    if "Secteur" in df_csv.columns:
        secteurs_hint = df_csv.set_index("Ticker")["Secteur"].to_dict()
elif tickers_text:
    tickers = [t.strip() for t in tickers_text.split(",") if t.strip()]

run = st.button("Lancer l’analyse", type="primary", disabled=(len(tickers) == 0))

# =============================
# Exécution + Affichage + Export
# =============================
if run:
    progress = st.progress(0.0, text="Analyse en cours…")
    results: List[Dict] = []
    total = len(tickers)

    for i, tk in enumerate(tickers, start=1):
        st.write(f"• Analyse de **{tk}** ({i}/{total})")
        row = analyze_ticker(tk, secteurs_hint, debug=debug)
        results.append(row)
        progress.progress(i / total, text=f"Analyse de {tk} ({i}/{total})")
        if sleep_sec and sleep_sec > 0:
            time.sleep(float(sleep_sec))

    df_results = pd.DataFrame(results)

    st.subheader("Résultats")
    st.dataframe(df_results, width="stretch")

    # Export Excel (en mémoire)
    output_name = "Scldm.xlsx" if len(tickers) > 1 else f"{tickers[0]}_analyse.xlsx"
    buffer = io.BytesIO()
    with pd.ExcelWriter(buffer, engine="openpyxl") as writer:
        df_results.to_excel(writer, index=False, sheet_name="Analyse")

    st.download_button(
        label="⬇️ Télécharger l’Excel",
        data=buffer.getvalue(),
        file_name=output_name,
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    )

    st.success("Analyse terminée.")
