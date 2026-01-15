import streamlit as st
import pandas as pd
import requests
import json
from datetime import datetime

# --------------------------------------------------
# GLOBAL SETTINGS
# --------------------------------------------------
API_KEY = st.secrets.get("BLS_API_KEY", "")
BLS_URL = "https://api.bls.gov/publicAPI/v2/timeseries/data/"

st.set_page_config(page_title="US Macro Dashboard", layout="wide")
st.title("📊 US Macro Dashboard (BLS Data)")
st.write("Choose a dataset from the sidebar and click **Run**.")

# -------------------------------
# BLS disruption warning helper
# -------------------------------
def bls_disruption_warning(message: str):
    st.warning(
        f"**Data disruption notice**\n\n{message}",
        icon="⚠️"
    )
#---------------------------------------------------
# CHARTING
#---------------------------------------------------
def add_charts(chart_df: pd.DataFrame, title: str):
    """
    chart_df: index must be datetime (monthly), columns are series to plot
    """
    if chart_df is None or chart_df.empty:
        st.info("No data available to chart.")
        return

    st.markdown(f"### {title} (chart)")
    st.line_chart(chart_df)


# --------------------------------------------------
# GENERIC FETCH FUNCTION
# --------------------------------------------------
def fetch_bls(payload):
    response = requests.post(
        BLS_URL,
        data=json.dumps(payload),
        headers={"Content-Type": "application/json"},
        timeout=30
    )
    data = response.json()
    if data.get("status") != "REQUEST_SUCCEEDED":
        st.error("❌ BLS API Error")
        return None
    return data


# --------------------------------------------------
# HELPER: FETCH SINGLE SERIES (WITH API KEY)
# --------------------------------------------------
def fetch_bls_series(series_id: str) -> pd.Series:
    payload = {
        "seriesid": [series_id],
        "registrationkey": API_KEY
    }

    data = fetch_bls(payload)
    if data is None:
        return None

    obs = data["Results"]["series"][0]["data"]
    df = pd.DataFrame(obs)
    df = df[df["period"].str.startswith("M")].copy()

    df["year"] = df["year"].astype(int)
    df["month"] = df["period"].str[1:].astype(int)
    df["date"] = pd.to_datetime(dict(year=df["year"], month=df["month"], day=1))
    df["value"] = pd.to_numeric(df["value"], errors="coerce")

    return df.set_index("date")["value"].sort_index()

# ---------------------------------------------------
#  Supercore
# --------------------------------------------------

REL_IMPORTANCE_URL = "https://www.bls.gov/cpi/tables/relative-importance/2024.htm"

@st.cache_data(ttl=60*60*24)  # cache for 24h
def get_supercore_weights():
    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120 Safari/537.36"
        ),
        "Referer": "https://www.bls.gov/cpi/"
    }
    r = requests.get(REL_IMPORTANCE_URL, headers=headers, timeout=30)
    r.raise_for_status()

    t = pd.read_html(r.text)[0]
    t.columns = [str(c).strip() for c in t.columns]
    item_col = t.columns[0]
    cpiu_col = next(c for c in t.columns if "CPI-U" in c)

    t[item_col] = t[item_col].astype(str).str.strip()
    t[cpiu_col] = pd.to_numeric(t[cpiu_col], errors="coerce")

    def pick(label):
        hit = t[t[item_col].str.contains(label, case=False, na=False)]
        val = hit.iloc[0][cpiu_col]
        return float(val) / 100.0

    w_cs   = pick("Services less energy services")
    w_rent = pick("Rent of primary residence")
    w_oer  = pick("Owners' equivalent rent of residences")

    return w_cs, w_rent, w_oer


def fetch_bls_df(series_map: dict, years_back: int = 5) -> pd.DataFrame:
    """
    series_map: {series_id: friendly_name}
    years_back: how many years of history to pull (rolling)
    returns df with columns: Date + friendly_name columns
    """
    endyear = datetime.today().year
    startyear = endyear - years_back

    payload = {
        "seriesid": list(series_map.keys()),
        "startyear": str(startyear),
        "endyear": str(endyear),
        "registrationkey": API_KEY
    }

    data = fetch_bls(payload)
    if data is None:
        return pd.DataFrame()

    dfs = []
    for s in data["Results"]["series"]:
        sid = s["seriesID"]
        name = series_map[sid]

        df = pd.DataFrame(s["data"])
        df = df[df["period"].str.startswith("M")].copy()

        df["Date"] = pd.to_datetime(
            df["year"] + "-" + df["period"].str.replace("M", ""),
            format="%Y-%m",
            errors="coerce"
        )
        df[name] = pd.to_numeric(df["value"], errors="coerce")
        df = df[["Date", name]].sort_values("Date")
        dfs.append(df)

    out = dfs[0]
    for d in dfs[1:]:
        out = out.merge(d, on="Date", how="outer")
    return out.sort_values("Date")



# --------------------------------------------------
# CPI 3dp
# --------------------------------------------------
def run_cpi_3dp():
    bls_disruption_warning(
        "- October m/m CPI was not released; short-term momentum may be distorted around this period.\n"
    )

    # SA for m/m + NSA for y/y
    series = {
        "CUSR0000SA0": "Headline CPI SA",
        "CUSR0000SA0L1E": "Core CPI SA",
        "CUUR0000SA0": "Headline CPI NSA",
        "CUUR0000SA0L1E": "Core CPI NSA",
    }

    payload = {
        "seriesid": list(series.keys()),
        "startyear": "2023",   # need 12m history for y/y
        "endyear": "2025",
        "registrationkey": API_KEY
    }

    data = fetch_bls(payload)
    if data is None:
        return

    # Build one combined dataframe of index levels
    dfs = []
    for s in data["Results"]["series"]:
        sid = s["seriesID"]
        name = series[sid]
        df = pd.DataFrame(s["data"])
        df = df[df["period"].str.startswith("M")].copy()

        df["Date"] = pd.to_datetime(
            df["year"] + "-" + df["period"].str.replace("M", ""),
            format="%Y-%m",
            errors="coerce"
        )
        df[name] = pd.to_numeric(df["value"], errors="coerce")
        df = df[["Date", name]].sort_values("Date")
        dfs.append(df)

    out = dfs[0]
    for d in dfs[1:]:
        out = out.merge(d, on="Date", how="outer")

    out = out.sort_values("Date")

    # ---- Compute changes ----
    # m/m from SA (continuity-safe: blanks if prior month missing)
    out["Headline CPI m/m"] = ((out["Headline CPI SA"] / out["Headline CPI SA"].shift(1) - 1) * 100)
    out.loc[out["Headline CPI SA"].isna() | out["Headline CPI SA"].shift(1).isna(), "Headline CPI m/m"] = pd.NA

    out["Core CPI m/m"] = ((out["Core CPI SA"] / out["Core CPI SA"].shift(1) - 1) * 100)
    out.loc[out["Core CPI SA"].isna() | out["Core CPI SA"].shift(1).isna(), "Core CPI m/m"] = pd.NA

    # y/y from NSA
    out["Headline CPI y/y"] = ((out["Headline CPI NSA"] / out["Headline CPI NSA"].shift(12) - 1) * 100)
    out["Core CPI y/y"] = ((out["Core CPI NSA"] / out["Core CPI NSA"].shift(12) - 1) * 100)

    # Keep just the 4 prints
    final = out[["Date", "Headline CPI m/m", "Core CPI m/m", "Headline CPI y/y", "Core CPI y/y"]]

    # Last 12 months, newest first
    final = final.sort_values("Date", ascending=False).head(12).copy()
    final["Date"] = final["Date"].dt.strftime("%Y-%m")

    # Format to always show 3dp (no % sign)
    for col in ["Headline CPI m/m", "Core CPI m/m", "Headline CPI y/y", "Core CPI y/y"]:
        final[col] = final[col].apply(lambda x: "" if pd.isna(x) else f"{x:.3f}")

    st.subheader("CPI (m/m and y/y, 3dp)")
    st.dataframe(final, use_container_width=True)

    # -----------------------------
    # Charts (m/m and y/y separate)
    # -----------------------------
    chart_df = out.set_index("Date").sort_index()
    
    mm_cols = ["Headline CPI m/m", "Core CPI m/m"]
    yy_cols = ["Headline CPI y/y", "Core CPI y/y"]
    
    st.markdown("### CPI m/m (SA) – chart")
    st.line_chart(chart_df[mm_cols])
    
    st.markdown("### CPI y/y (NSA) – chart")
    st.line_chart(chart_df[yy_cols])




# --------------------------------------------------
# NFP & Unemployment Rate
# --------------------------------------------------
def run_nfp():
    bls_disruption_warning(
    "- The October Unemployment rate was never released.\n"
)
    series_ids = {
        "CES0000000001": "Nonfarm Payrolls Level",
        "LNS14000000": "Unemployment Rate"
    }

    payload = {
        "seriesid": list(series_ids.keys()),
        "startyear": "2010",
        "endyear": "2025",
        "registrationkey": API_KEY
    }

    data = fetch_bls(payload)
    if data is None:
        return

    dfs = []

    for s in data["Results"]["series"]:
        sid = s["seriesID"]
        name = series_ids[sid]

        rows = []
        for e in s["data"]:
            if not e["period"].startswith("M"):
                continue

            rows.append({
                "date": f"{e['year']}-{e['period'][1:]}-01",
                name: pd.to_numeric(e["value"], errors="coerce")
            })

        df = pd.DataFrame(rows)
        df["date"] = pd.to_datetime(df["date"])
        df = df.sort_values("date")
        dfs.append(df)

    # Merge level + unemployment
    df = dfs[0]
    for extra in dfs[1:]:
        df = df.merge(extra, on="date", how="outer")

    # --------------------------------------------------
    # Calculate NFP m/m change (thousands)
    # --------------------------------------------------
    df["NFP m/m change"] = df["Nonfarm Payrolls Level"].diff()

    # Rolling averages of the CHANGE
    df["NFP 3m avg"] = df["NFP m/m change"].rolling(3).mean()
    df["NFP 6m avg"] = df["NFP m/m change"].rolling(6).mean()
    df["NFP 12m avg"] = df["NFP m/m change"].rolling(12).mean()

    # Clean up
    df = df.drop(columns=["Nonfarm Payrolls Level"])
    df = df.round(1)

    # Latest first
    df = df.sort_values("date", ascending=False).head(24)

    # Remove timestamp
    df["date"] = df["date"].dt.strftime("%Y-%m")
    df = df.set_index("date")

    st.subheader("Nonfarm Payrolls (m/m change, K) & Unemployment Rate %")
    st.dataframe(df, use_container_width=True)




# --------------------------------------------------
# CPI Core Goods & Services (fixed)
# --------------------------------------------------
def run_cpi_goods_services():
    # (optional) your warning call
    # bls_disruption_warning("...")

    # --- Your existing SA/NSA pulls via fetch_bls_series (kept) ---
    series_ids = {
        "Core goods": ("CUSR0000SACL1E", "CUUR0000SACL1E"),
        "Core services": ("CUSR0000SASLE", "CUUR0000SASLE")
    }

    tables = {}
    for label, (sa_id, nsa_id) in series_ids.items():
        sa = fetch_bls_series(sa_id)
        nsa = fetch_bls_series(nsa_id)

        if sa is None or nsa is None:
            st.error(f"❌ No data returned for: {label}")
            return

        df = pd.DataFrame(index=sa.index)
        df["m/m"] = (sa / sa.shift(1) - 1) * 100
        df["y/y"] = (nsa / nsa.shift(12) - 1) * 100
        tables[label] = df

    combined = pd.concat(tables, axis=1)  # <-- IMPORTANT: no dropna()

    # --- Supercore add-on ---
    # SA series (m/m)
    SA_MAP = {
        "CUSR0000SASLE": "Core Services SA",
        "CUSR0000SEHA":  "Rent SA",
        "CUSR0000SEHC":  "OER SA"
    }

    # NSA series (y/y)
    NSA_MAP = {
        "CUUR0000SASLE": "Core Services NSA",
        "CUUR0000SEHA":  "Rent NSA",
        "CUUR0000SEHC":  "OER NSA"
    }

    df_sa = fetch_bls_df(SA_MAP)
    df_nsa = fetch_bls_df(NSA_MAP)

    if not df_sa.empty and not df_nsa.empty:
        sc = df_sa.merge(df_nsa, on="Date", how="outer").sort_values("Date")

        W_CS, W_RENT, W_OER = get_supercore_weights()
        DEN = W_CS - W_RENT - W_OER

        sc["Supercore Index SA"] = (
            W_CS * sc["Core Services SA"]
            - W_RENT * sc["Rent SA"]
            - W_OER * sc["OER SA"]
        ) / DEN

        sc["Supercore Index NSA"] = (
            W_CS * sc["Core Services NSA"]
            - W_RENT * sc["Rent NSA"]
            - W_OER * sc["OER NSA"]
        ) / DEN

        sc["Supercore m/m"] = (sc["Supercore Index SA"] / sc["Supercore Index SA"].shift(1) - 1) * 100
        sc["Supercore y/y"] = (sc["Supercore Index NSA"] / sc["Supercore Index NSA"].shift(12) - 1) * 100

        sc_out = sc[["Date", "Supercore m/m", "Supercore y/y"]].set_index("Date")
        # merge into combined (aligned on Date index)
        sc_out = sc[["Date", "Supercore m/m", "Supercore y/y"]].set_index("Date")

        # Keep supercore aligned to the same last 12 months as the table view
        sc_out = sc_out.tail(12)

        combined = pd.concat([combined, sc_out], axis=1)


    # --- Last 12 months, latest first ---
    last12 = combined.iloc[-12:].iloc[::-1].round(2)
    last12.index = last12.index.strftime("%B %Y")
    last12.index.name = "Month"

    # Flatten MultiIndex columns (Headline m/m etc.)
    flat_cols = []
    for col in last12.columns:
        if isinstance(col, tuple):
            flat_cols.append(f"{col[0]} {col[1]}")
        else:
            flat_cols.append(col)
    last12.columns = flat_cols

    st.subheader("CPI Core Goods & Services (m/m SA, y/y NSA) + Supercore (Core Services ex-Rent+OER)")
    st.dataframe(last12, use_container_width=True)

    # -----------------------------
    # Charts (m/m and y/y separate)
    # -----------------------------
    chart_df = combined.copy()
    
    # combined has MultiIndex columns for core goods/services, but Supercore is single-level
    mm = pd.DataFrame(index=chart_df.index)
    yy = pd.DataFrame(index=chart_df.index)
    
    # Core goods/services (MultiIndex)
    mm["Core goods"] = chart_df[("Core goods", "m/m")]
    mm["Core services"] = chart_df[("Core services", "m/m")]
    yy["Core goods"] = chart_df[("Core goods", "y/y")]
    yy["Core services"] = chart_df[("Core services", "y/y")]
    
    # Supercore (single level)
    if "Supercore m/m" in chart_df.columns:
        mm["Supercore"] = chart_df["Supercore m/m"]
    if "Supercore y/y" in chart_df.columns:
        yy["Supercore"] = chart_df["Supercore y/y"]
    
    # Sort oldest -> newest for charts
    mm = mm.sort_index()
    yy = yy.sort_index()
    
    st.markdown("### Core goods/services + Supercore m/m (SA) – chart")
    st.line_chart(mm)
    
    st.markdown("### Core goods/services + Supercore y/y (NSA) – chart")
    st.line_chart(yy)


# --------------------------------------------------
# Annualised CPI (3m & 6m)
# --------------------------------------------------
def run_cpi_annualised():
    bls_disruption_warning(
        "- October and November 2025 m/m CPI prints were not released.\n"
        "- Short-term annualised CPI around late-2025 may be distorted by missing index levels.\n"
    )

    series_ids = {
        "CUSR0000SA0": "Headline CPI SA",
        "CUSR0000SA0L1E": "Core CPI SA"
    }

    payload = {
        "seriesid": list(series_ids.keys()),
        "startyear": "2015",
        "endyear": "2025",
        "registrationkey": API_KEY
    }

    data = fetch_bls(payload)
    if data is None:
        return

    dfs = []

    # Convert BLS data → tidy DataFrames
    for s in data["Results"]["series"]:
        sid = s["seriesID"]
        name = series_ids[sid]

        rows = []
        for item in s["data"]:
            period = item["period"]
            if not period.startswith("M"):
                continue

            year = item["year"]
            month = period[1:]

            rows.append({
                "date": f"{year}-{month}-01",
                name: pd.to_numeric(item["value"], errors="coerce")
            })

        df = pd.DataFrame(rows)
        df["date"] = pd.to_datetime(df["date"])
        df = df.sort_values("date")
        dfs.append(df)

    # Merge Headline + Core
    cpi = dfs[0]
    for extra in dfs[1:]:
        cpi = cpi.merge(extra, on="date", how="outer")

    # --------------------------------------------------
    # Make a complete monthly index (so October 2025 exists as NaN)
    # --------------------------------------------------
    cpi = cpi.sort_values("date")
    full_range = pd.date_range(
        start=cpi["date"].min(),
        end=cpi["date"].max(),
        freq="MS"          # Month start frequency
    )

    cpi = (
        cpi
        .set_index("date")
        .reindex(full_range)
        .rename_axis("date")
        .reset_index()
    )

    # --------------------------------------------------
    # Rename for clarity
    # --------------------------------------------------
    cpi = cpi.rename(columns={
        "Headline CPI SA": "Headline CPI Index SA",
        "Core CPI SA": "Core CPI Index SA"
    })

    # --------------------------------------------------
    # Annualised CPI calculations with continuity checks
    # --------------------------------------------------
    for col in ["Headline CPI Index SA", "Core CPI Index SA"]:
        # Base ratios
        ratio_3m = cpi[col] / cpi[col].shift(3)
        ratio_6m = cpi[col] / cpi[col].shift(6)

        # Require a full, uninterrupted history in the window:
        # 3m ann: need t, t-1, t-2, t-3 all non-NaN
        valid_3m = (
            cpi[col].notna()
            & cpi[col].shift(1).notna()
            & cpi[col].shift(2).notna()
            & cpi[col].shift(3).notna()
        )

        # 6m ann: need t, t-1, ..., t-6 all non-NaN
        valid_6m = (
            cpi[col].notna()
            & cpi[col].shift(1).notna()
            & cpi[col].shift(2).notna()
            & cpi[col].shift(3).notna()
            & cpi[col].shift(4).notna()
            & cpi[col].shift(5).notna()
            & cpi[col].shift(6).notna()
        )

        ann_3m = ((ratio_3m ** 4) - 1) * 100
        ann_6m = ((ratio_6m ** 2) - 1) * 100

        # Drop values where the window is broken by missing months
        ann_3m[~valid_3m] = pd.NA
        ann_6m[~valid_6m] = pd.NA

        cpi[f"{col} 3m ann"] = ann_3m
        cpi[f"{col} 6m ann"] = ann_6m

    # Round annualised values
    ann_cols = [col for col in cpi.columns if "ann" in col]
    cpi[ann_cols] = cpi[ann_cols].astype("float").round(3)

    # --------------------------------------------------
    # Prepare display table
    # --------------------------------------------------
    cpi = cpi.sort_values("date", ascending=False).head(12)

    cpi["date"] = cpi["date"].dt.strftime("%Y-%m")
    cpi = cpi.set_index("date")

    st.subheader("Annualised CPI (3m & 6m %)")
    st.dataframe(cpi, use_container_width=True)


# --------------------------------------------------
# PPI → PCE Components
# --------------------------------------------------
def run_ppi_pce():
    
    ids = {
        "PCU5239405239401": "Portfolio Management PPI",
        "WPS3022": "Passenger Airline Services",
        "WPS511101": "Physician Care",
        "WPS511103": "Home Health & Hospice Care",
        "WPS511104": "Hospital Outpatient Care",
        "WPS512101": "Hospital Inpatient Care",
        "WPS512102": "Nursing Home Care"
    }

    payload = {
        "seriesid": list(ids.keys()),
        "startyear": "2023",
        "endyear": "2025",
        "registrationkey": API_KEY
    }

    data = fetch_bls(payload)
    if data is None:
        return

    dfs = []

    for s in data["Results"]["series"]:
        sid = s["seriesID"]
        name = ids[sid]

        df = pd.DataFrame(s["data"])
        df["Date"] = pd.to_datetime(
            df["year"] + "-" + df["period"].str.replace("M", ""),
            errors="coerce"
        )
        df["Value"] = pd.to_numeric(df["value"], errors="coerce")
        df = df.sort_values("Date")
        df[name] = df["Value"].pct_change() * 100

        dfs.append(df[["Date", name]])

    final = dfs[0]
    for x in dfs[1:]:
        final = final.merge(x, on="Date", how="outer")

    final = final.sort_values("Date", ascending=False).set_index("Date").round(2)

    # Fix timestamp formatting
    final.index = final.index.strftime("%Y-%m")

    st.subheader("PPI → PCE Components (m/m %)")
    st.dataframe(final.head(24), use_container_width=True)

    # --------------------------------------------------
    # Headline-style summary box (latest vs previous)
    # --------------------------------------------------
    display_df = final.head(24)

    if display_df.empty:
        st.info("No PPI → PCE component data available to summarise yet.")
    else:
        latest = display_df.iloc[0]
        prev = display_df.iloc[1] if len(display_df) > 1 else None

        headline_lines = []
        for col in display_df.columns:
            latest_val = latest[col]
            prev_val = prev[col] if prev is not None else pd.NA

            if pd.isna(latest_val):
                continue

            if prev is not None and not pd.isna(prev_val):
                headline_lines.append(
                    f"{col}: {latest_val:.2f}% (prev. {prev_val:.2f}%)"
                )
            else:
                headline_lines.append(f"{col}: {latest_val:.2f}%")

        headline_text = "\n".join(headline_lines)

        st.markdown("**PPI → PCE Components (headline format)**")
        st.text_area("", value=headline_text, height=200)





# --------------------------------------------------
# JOLTS
# --------------------------------------------------
def run_jolts():

    series_ids = {
        "Headline JOLTS": "JTS000000000000000JOL",
        "Vacancy Rate": "JTS000000000000000JOR",
        "Quits": "JTS000000000000000QUL",
        "Quits Rate": "JTS000000000000000QUR",
        "Separations": "JTS000000000000000TSL",
        "Separations rate": "JTS000000000000000TSR",
        "Hires": "JTS000000000000000HIL",
        "Hire rate": "JTS000000000000000HIR",
        "Layoffs and discharges": "JTS000000000000000LDL",
        "Layoffs and discharges rate": "JTS000000000000000LDR"
    }

    payload = {
        "seriesid": list(series_ids.values()),
        "startyear": "2015",
        "endyear": "2025",
        "registrationkey": API_KEY
    }

    data = fetch_bls(payload)
    if data is None:
        return

    rows = []
    for s in data["Results"]["series"]:
        sid = s["seriesID"]
        name = [k for k, v in series_ids.items() if v == sid][0]

        for e in s["data"]:
            if e["period"].startswith("M"):
                rows.append({
                    "Date": f"{e['year']}-{e['period'][1:]}",
                    "Series": name,
                    "Value": pd.to_numeric(e["value"], errors="coerce")

                })

    df = pd.DataFrame(rows)
    df = df.pivot(index="Date", columns="Series", values="Value")

    # Column order match
    df = df.reindex(columns=list(series_ids.keys()))

    # Newest → oldest
    df = df.sort_index(ascending=False)

    st.subheader("JOLTS Data (Levels in thousands, Rates in %)")
    st.dataframe(df.round(2), use_container_width=True)



# --------------------------------------------------
# SIDEBAR SELECTION
# --------------------------------------------------
st.sidebar.header("Select Dataset")

choice = st.sidebar.radio(
    "Choose:",
    [
        "CPI (m/m, 3dp)",
        "CPI Core Goods & Services",
        "Annualised CPI (3m & 6m)",
        "PPI → PCE Components",
        "JOLTS",
        "NFP & Unemployment"
    ]
)


if st.sidebar.button("Run"):
    if choice == "CPI (m/m, 3dp)":
        run_cpi_3dp()
    elif choice == "CPI Core Goods & Services":
        run_cpi_goods_services()
    elif choice == "Annualised CPI (3m & 6m)":
        run_cpi_annualised()
    elif choice == "PPI → PCE Components":
        run_ppi_pce()
    elif choice == "JOLTS":
        run_jolts()
    elif choice == "NFP & Unemployment":
        run_nfp()
