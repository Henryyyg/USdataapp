import streamlit as st
import pandas as pd
import requests
import json
import io
from datetime import datetime

# --------------------------------------------------
# GLOBAL SETTINGS
# --------------------------------------------------
API_KEY = st.secrets.get("BLS_API_KEY", "")
BLS_URL = "https://api.bls.gov/publicAPI/v2/timeseries/data/"

st.set_page_config(page_title="US Macro Dashboard", layout="wide")
st.title("📊 US Macro Dashboard (BLS Data)")
st.write("Choose a dataset from the sidebar and click **Run**.")
CURRENT_YEAR = datetime.today().year

# -------------------------------
# BLS disruption warning helper
# -------------------------------
def bls_disruption_warning(message: str):
    st.warning(
        f"**Data disruption notice**\n\n{message}",
        icon="⚠️"
    )

# --------------------------------------------------
# CHARTING
# --------------------------------------------------
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
    try:
        response = requests.post(
            BLS_URL,
            data=json.dumps(payload),
            headers={"Content-Type": "application/json"},
            timeout=30
        )
        response.raise_for_status()
        data = response.json()

        if data.get("status") != "REQUEST_SUCCEEDED":
            st.error(f"❌ BLS API Error: {data.get('message', 'Unknown error')}")
            return None

        return data

    except Exception as e:
        st.error(f"❌ Error fetching BLS data: {e}")
        return None

# --------------------------------------------------
# HELPER: FETCH SINGLE SERIES
# --------------------------------------------------
def fetch_bls_series(series_id: str, years_back: int = 5) -> pd.Series:
    endyear = datetime.today().year
    startyear = endyear - years_back

    payload = {
        "seriesid": [series_id],
        "startyear": str(startyear),
        "endyear": str(endyear),
        "registrationkey": API_KEY
    }

    data = fetch_bls(payload)
    if data is None:
        return None

    series_list = data.get("Results", {}).get("series", [])
    if not series_list:
        return None

    obs = series_list[0].get("data", [])
    if not obs:
        return None

    df = pd.DataFrame(obs)
    df = df[df["period"].str.startswith("M")].copy()

    df["year"] = df["year"].astype(int)
    df["month"] = df["period"].str[1:].astype(int)
    df["date"] = pd.to_datetime(dict(year=df["year"], month=df["month"], day=1))
    df["value"] = pd.to_numeric(df["value"], errors="coerce")

    return df.set_index("date")["value"].sort_index()

# --------------------------------------------------
# HELPER: FETCH MULTIPLE SERIES INTO ONE DATAFRAME
# --------------------------------------------------
def fetch_bls_df(series_map: dict, years_back: int = 5) -> pd.DataFrame:
    """
    series_map: {series_id: friendly_name}
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

    results = data.get("Results", {}).get("series", [])
    if not results:
        return pd.DataFrame()

    dfs = []
    for s in results:
        sid = s["seriesID"]
        if sid not in series_map:
            continue

        name = series_map[sid]
        df = pd.DataFrame(s["data"])
        df = df[df["period"].str.startswith("M")].copy()

        df["Date"] = pd.to_datetime(
            df["year"] + "-" + df["period"].str.replace("M", "", regex=False),
            format="%Y-%m",
            errors="coerce"
        )
        df[name] = pd.to_numeric(df["value"], errors="coerce")
        df = df[["Date", name]].sort_values("Date")
        dfs.append(df)

    if not dfs:
        return pd.DataFrame()

    out = dfs[0]
    for d in dfs[1:]:
        out = out.merge(d, on="Date", how="outer")

    return out.sort_values("Date")

# ---------------------------------------------------
# SUPERCORE WEIGHTS
# ---------------------------------------------------
@st.cache_data(ttl=60 * 60 * 24)
def get_supercore_weights():
    """
    Try BLS current workbook first.
    If blocked, fall back to fixed weights.
    Returns decimals.
    """
    url = "https://www.bls.gov/web/cpi/cpi-relative-importance.xlsx"

    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/122.0.0.0 Safari/537.36"
        ),
        "Referer": "https://www.bls.gov/cpi/"
    }

    try:
        r = requests.get(url, headers=headers, timeout=30)
        r.raise_for_status()

        raw = pd.read_excel(io.BytesIO(r.content), sheet_name=0, header=None)

        t = raw.iloc[:, [1, 2]].copy()
        t.columns = ["Item", "CPI-U"]

        t["Item"] = t["Item"].astype(str).str.strip()
        t["CPI-U"] = pd.to_numeric(t["CPI-U"], errors="coerce")
        t = t.dropna(subset=["CPI-U"])
        t = t[t["Item"] != ""].copy()

        def pick_exact(label: str):
            hit = t.loc[t["Item"].str.lower() == label.lower(), "CPI-U"]
            if hit.empty:
                return None
            return float(hit.iloc[0]) / 100.0

        w_cs = pick_exact("Services less energy services")
        w_rent = pick_exact("Rent of primary residence")
        w_oer = pick_exact("Owners' equivalent rent of residences")

        if w_cs is not None and w_rent is not None and w_oer is not None:
            return w_cs, w_rent, w_oer

    except Exception:
        pass

    # Fallback weights
    FALLBACK_W_CS = 0.607
    FALLBACK_W_RENT = 0.076
    FALLBACK_W_OER = 0.266

    return FALLBACK_W_CS, FALLBACK_W_RENT, FALLBACK_W_OER

# --------------------------------------------------
# CPI 3dp
# --------------------------------------------------
def run_cpi_3dp():
    bls_disruption_warning(
        "- October m/m CPI was not released; short-term momentum may be distorted around this period.\n"
    )

    series = {
        "CUSR0000SA0": "Headline CPI SA",
        "CUSR0000SA0L1E": "Core CPI SA",
        "CUUR0000SA0": "Headline CPI NSA",
        "CUUR0000SA0L1E": "Core CPI NSA",
    }

    payload = {
        "seriesid": list(series.keys()),
        "startyear": str(CURRENT_YEAR - 2),
        "endyear": str(CURRENT_YEAR),
        "registrationkey": API_KEY
    }

    data = fetch_bls(payload)
    if data is None:
        return

    dfs = []
    for s in data["Results"]["series"]:
        sid = s["seriesID"]
        name = series[sid]
        df = pd.DataFrame(s["data"])
        df = df[df["period"].str.startswith("M")].copy()

        df["Date"] = pd.to_datetime(
            df["year"] + "-" + df["period"].str.replace("M", "", regex=False),
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

    out["Headline CPI m/m"] = ((out["Headline CPI SA"] / out["Headline CPI SA"].shift(1) - 1) * 100)
    out.loc[out["Headline CPI SA"].isna() | out["Headline CPI SA"].shift(1).isna(), "Headline CPI m/m"] = pd.NA

    out["Core CPI m/m"] = ((out["Core CPI SA"] / out["Core CPI SA"].shift(1) - 1) * 100)
    out.loc[out["Core CPI SA"].isna() | out["Core CPI SA"].shift(1).isna(), "Core CPI m/m"] = pd.NA

    out["Headline CPI y/y"] = ((out["Headline CPI NSA"] / out["Headline CPI NSA"].shift(12) - 1) * 100)
    out["Core CPI y/y"] = ((out["Core CPI NSA"] / out["Core CPI NSA"].shift(12) - 1) * 100)

    final = out[["Date", "Headline CPI m/m", "Core CPI m/m", "Headline CPI y/y", "Core CPI y/y"]]
    final = final.sort_values("Date", ascending=False).head(12).copy()
    final["Date"] = final["Date"].dt.strftime("%Y-%m")

    for col in ["Headline CPI m/m", "Core CPI m/m", "Headline CPI y/y", "Core CPI y/y"]:
        final[col] = final[col].apply(lambda x: "" if pd.isna(x) else f"{x:.3f}")

    st.subheader("CPI (m/m and y/y, 3dp)")
    st.dataframe(final, use_container_width=True)

    chart_df = out.set_index("Date").sort_index()

    mm_cols = ["Headline CPI m/m", "Core CPI m/m"]
    yy_cols = ["Headline CPI y/y", "Core CPI y/y"]

    st.markdown("### CPI m/m (SA) – chart")
    st.line_chart(chart_df[mm_cols])

    st.markdown("### CPI y/y (NSA) – chart")
    st.line_chart(chart_df[yy_cols])

    # --------------------------------------------------
    # CPI headline generator
    # --------------------------------------------------
    headline_df = out[
        [
            "Date",
            "Headline CPI m/m",
            "Core CPI m/m",
            "Headline CPI y/y",
            "Core CPI y/y"
        ]
    ].sort_values("Date", ascending=False).head(2)

    if len(headline_df) >= 2:
        latest = headline_df.iloc[0]
        prev = headline_df.iloc[1]

        def fmt3(value):
            return "N/A" if pd.isna(value) else f"{value:.3f}%"

        headline_text = (
            f"Headline CPI M/M: {fmt3(latest['Headline CPI m/m'])} "
            f"(prev. {fmt3(prev['Headline CPI m/m'])})\n"
            f"Core CPI M/M: {fmt3(latest['Core CPI m/m'])} "
            f"(prev. {fmt3(prev['Core CPI m/m'])})\n"
            f"Headline CPI Y/Y: {fmt3(latest['Headline CPI y/y'])} "
            f"(prev. {fmt3(prev['Headline CPI y/y'])})\n"
            f"Core CPI Y/Y: {fmt3(latest['Core CPI y/y'])} "
            f"(prev. {fmt3(prev['Core CPI y/y'])})"
        )

        st.markdown("### CPI headline format")
        st.text_area(
            "",
            value=headline_text,
            height=150,
            key="cpi_headline"
        )

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
        "startyear": str(CURRENT_YEAR - 2),
        "endyear": str(CURRENT_YEAR),
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

    df = dfs[0]
    for extra in dfs[1:]:
        df = df.merge(extra, on="date", how="outer")

    df["NFP m/m change"] = df["Nonfarm Payrolls Level"].diff()
    df["NFP 3m avg"] = df["NFP m/m change"].rolling(3).mean()
    df["NFP 6m avg"] = df["NFP m/m change"].rolling(6).mean()
    df["NFP 12m avg"] = df["NFP m/m change"].rolling(12).mean()

    df = df.drop(columns=["Nonfarm Payrolls Level"])
    df = df.round(1)

    df = df.sort_values("date", ascending=False).head(24)
    df["date"] = df["date"].dt.strftime("%Y-%m")
    df = df.set_index("date")

    st.subheader("Nonfarm Payrolls (m/m change, K) & Unemployment Rate %")
    st.dataframe(df, use_container_width=True)

    # --------------------------------------------------
    # NFP & Unemployment headline generator
    # --------------------------------------------------
    if len(df) >= 2:
        latest = df.iloc[0]
        prev = df.iloc[1]

        def fmt_jobs(value):
            return "N/A" if pd.isna(value) else f"{value:.1f}k"

        def fmt_rate(value):
            return "N/A" if pd.isna(value) else f"{value:.1f}%"

        headline_text = (
            f"NFP M/M Change: {fmt_jobs(latest['NFP m/m change'])} "
            f"(prev. {fmt_jobs(prev['NFP m/m change'])})\n"
            f"NFP 3M Average: {fmt_jobs(latest['NFP 3m avg'])} "
            f"(prev. {fmt_jobs(prev['NFP 3m avg'])})\n"
            f"NFP 6M Average: {fmt_jobs(latest['NFP 6m avg'])} "
            f"(prev. {fmt_jobs(prev['NFP 6m avg'])})\n"
            f"NFP 12M Average: {fmt_jobs(latest['NFP 12m avg'])} "
            f"(prev. {fmt_jobs(prev['NFP 12m avg'])})\n"
            f"Unemployment Rate: {fmt_rate(latest['Unemployment Rate'])} "
            f"(prev. {fmt_rate(prev['Unemployment Rate'])})"
        )

        st.markdown("### NFP & Unemployment headline format")
        st.text_area(
            "",
            value=headline_text,
            height=200,
            key="nfp_headline"
        )

# --------------------------------------------------
# CPI Core Goods & Services + Supercore
# --------------------------------------------------
def run_cpi_goods_services():
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

    combined = pd.concat(tables, axis=1)

    SA_MAP = {
        "CUSR0000SASLE": "Core Services SA",
        "CUSR0000SEHA": "Rent SA",
        "CUSR0000SEHC": "OER SA"
    }

    NSA_MAP = {
        "CUUR0000SASLE": "Core Services NSA",
        "CUUR0000SEHA": "Rent NSA",
        "CUUR0000SEHC": "OER NSA"
    }

    df_sa = fetch_bls_df(SA_MAP)
    df_nsa = fetch_bls_df(NSA_MAP)

    sc = None

    if not df_sa.empty and not df_nsa.empty:
        sc = df_sa.merge(df_nsa, on="Date", how="outer").sort_values("Date")

        W_CS, W_RENT, W_OER = get_supercore_weights()

        if W_CS is None or W_RENT is None or W_OER is None:
            st.warning("Supercore could not be calculated because BLS relative-importance weights were unavailable.")
            sc = None
        else:
            DEN = W_CS - W_RENT - W_OER

            if pd.isna(DEN) or abs(DEN) < 1e-12:
                st.error("Supercore denominator is invalid.")
                sc = None
            else:
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

                sc["Supercore m/m"] = (
                    sc["Supercore Index SA"] / sc["Supercore Index SA"].shift(1) - 1
                ) * 100

                sc["Supercore y/y"] = (
                    sc["Supercore Index NSA"] / sc["Supercore Index NSA"].shift(12) - 1
                ) * 100

                sc_out = sc[["Date", "Supercore m/m", "Supercore y/y"]].set_index("Date")
                sc_out = sc_out.tail(12)

                combined = pd.concat([combined, sc_out], axis=1)

    combined = combined.sort_index()
    last12 = combined.tail(12).iloc[::-1].round(2)
    last12.index = last12.index.strftime("%B %Y")
    last12.index.name = "Month"

    flat_cols = []
    for col in last12.columns:
        if isinstance(col, tuple):
            flat_cols.append(f"{col[0]} {col[1]}")
        else:
            flat_cols.append(col)
    last12.columns = flat_cols

    st.subheader("CPI Core Goods & Services (m/m SA, y/y NSA) + Supercore (Core Services ex-Rent+OER)")
    st.dataframe(last12, use_container_width=True)

    end = combined.sort_index().index.max()
    end = pd.Timestamp(end.year, end.month, 1)

    chart_index = pd.date_range(end=end, periods=12, freq="MS")

    chart_base = (
        combined.sort_index()
        .reindex(chart_index)
        .interpolate(method="linear", limit=1)
    )

    mm = pd.DataFrame(index=chart_base.index)
    yy = pd.DataFrame(index=chart_base.index)

    mm["Core goods"] = chart_base[("Core goods", "m/m")]
    mm["Core services"] = chart_base[("Core services", "m/m")]
    yy["Core goods"] = chart_base[("Core goods", "y/y")]
    yy["Core services"] = chart_base[("Core services", "y/y")]

    if "Supercore m/m" in chart_base.columns:
        mm["Supercore"] = chart_base["Supercore m/m"]
    if "Supercore y/y" in chart_base.columns:
        yy["Supercore"] = chart_base["Supercore y/y"]

    st.markdown("### Core goods/services + Supercore m/m (SA) – last 12 months")
    st.line_chart(mm)

    st.markdown("### Core goods/services + Supercore y/y (NSA) – last 12 months")
    st.line_chart(yy)

    st.caption("Note: chart lines interpolate across single missing months due to BLS disruptions (tables remain unfilled).")

    # --------------------------------------------------
    # Core Goods / Services / Supercore headline generator
    # --------------------------------------------------
    headline_df = last12.head(2)

    if len(headline_df) >= 2:
        latest = headline_df.iloc[0]
        prev = headline_df.iloc[1]

        metrics = [
            "Core goods m/m",
            "Core goods y/y",
            "Core services m/m",
            "Core services y/y",
            "Supercore m/m",
            "Supercore y/y"
        ]

        labels = {
            "Core goods m/m": "Core Goods CPI M/M",
            "Core goods y/y": "Core Goods CPI Y/Y",
            "Core services m/m": "Core Services CPI M/M",
            "Core services y/y": "Core Services CPI Y/Y",
            "Supercore m/m": "Supercore CPI M/M",
            "Supercore y/y": "Supercore CPI Y/Y"
        }

        def fmt2(value):
            return "N/A" if pd.isna(value) else f"{value:.2f}%"

        headline_lines = []

        for metric in metrics:
            if metric in headline_df.columns:
                headline_lines.append(
                    f"{labels[metric]}: {fmt2(latest[metric])} "
                    f"(prev. {fmt2(prev[metric])})"
                )

        st.markdown("### Core CPI components headline format")
        st.text_area(
            "",
            value="\n".join(headline_lines),
            height=220,
            key="core_components_headline"
        )

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
        "startyear": str(CURRENT_YEAR - 2),
        "endyear": str(CURRENT_YEAR),
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

    cpi = dfs[0]
    for extra in dfs[1:]:
        cpi = cpi.merge(extra, on="date", how="outer")

    cpi = cpi.sort_values("date")
    full_range = pd.date_range(
        start=cpi["date"].min(),
        end=cpi["date"].max(),
        freq="MS"
    )

    cpi = (
        cpi
        .set_index("date")
        .reindex(full_range)
        .rename_axis("date")
        .reset_index()
    )

    cpi = cpi.rename(columns={
        "Headline CPI SA": "Headline CPI Index SA",
        "Core CPI SA": "Core CPI Index SA"
    })

    for col in ["Headline CPI Index SA", "Core CPI Index SA"]:
        ratio_3m = cpi[col] / cpi[col].shift(3)
        ratio_6m = cpi[col] / cpi[col].shift(6)

        valid_3m = (
            cpi[col].notna()
            & cpi[col].shift(1).notna()
            & cpi[col].shift(2).notna()
            & cpi[col].shift(3).notna()
        )

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

        ann_3m[~valid_3m] = pd.NA
        ann_6m[~valid_6m] = pd.NA

        cpi[f"{col} 3m ann"] = ann_3m
        cpi[f"{col} 6m ann"] = ann_6m

    ann_cols = [col for col in cpi.columns if "ann" in col]
    cpi[ann_cols] = cpi[ann_cols].astype("float").round(3)

    cpi = cpi.sort_values("date", ascending=False).head(12)
    cpi["date"] = cpi["date"].dt.strftime("%Y-%m")
    cpi = cpi.set_index("date")

    st.subheader("Annualised CPI (3m & 6m %)")
    st.dataframe(cpi, use_container_width=True)

    # --------------------------------------------------
    # Annualised CPI headline generator
    # --------------------------------------------------
    if len(cpi) >= 2:
        latest = cpi.iloc[0]
        prev = cpi.iloc[1]

        metrics = [
            "Headline CPI Index SA 3m ann",
            "Headline CPI Index SA 6m ann",
            "Core CPI Index SA 3m ann",
            "Core CPI Index SA 6m ann"
        ]

        labels = {
            "Headline CPI Index SA 3m ann": "Headline CPI 3M Annualised",
            "Headline CPI Index SA 6m ann": "Headline CPI 6M Annualised",
            "Core CPI Index SA 3m ann": "Core CPI 3M Annualised",
            "Core CPI Index SA 6m ann": "Core CPI 6M Annualised"
        }

        def fmt3(value):
            return "N/A" if pd.isna(value) else f"{value:.3f}%"

        headline_lines = []

        for metric in metrics:
            if metric in cpi.columns:
                headline_lines.append(
                    f"{labels[metric]}: {fmt3(latest[metric])} "
                    f"(prev. {fmt3(prev[metric])})"
                )

        st.markdown("### Annualised CPI headline format")
        st.text_area(
            "",
            value="\n".join(headline_lines),
            height=180,
            key="annualised_cpi_headline"
        )

# --------------------------------------------------
# PPI → PCE Components
# --------------------------------------------------
def run_ppi_pce():
    ids = {
        "PCU5239405239401": "Portfolio Management PPI",
        "PCU4811114811111": "Air Passenger Transport PPI",
        "WPS511101": "Physician Care",
        "WPS511103": "Home Health & Hospice Care",
        "WPS511104": "Hospital Outpatient Care",
        "WPS512101": "Hospital Inpatient Care",
        "WPS512102": "Nursing Home Care"
    }

    payload = {
        "seriesid": list(ids.keys()),
        "startyear": str(CURRENT_YEAR - 2),
        "endyear": str(CURRENT_YEAR),
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
        df = df[df["period"].str.startswith("M")].copy()

        df["Date"] = pd.to_datetime(
            df["year"] + "-" + df["period"].str.replace("M", "", regex=False),
            format="%Y-%m",
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
    final.index = final.index.strftime("%Y-%m")

    st.subheader("PPI → PCE Components (m/m %)")
    st.dataframe(final.head(24), use_container_width=True)

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
        "startyear": str(CURRENT_YEAR - 2),
        "endyear": str(CURRENT_YEAR),
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
    df = df.reindex(columns=list(series_ids.keys()))
    df = df.sort_index(ascending=False)

    st.subheader("JOLTS Data (Levels in thousands, Rates in %)")
    st.dataframe(df.round(2), use_container_width=True)

# --------------------------------------------------
# CPI → PCE Components
# --------------------------------------------------
def run_cpi_pce():

    ids = {
        "CUSR0000SEHA": "Rent of Primary Residence",
        "CUSR0000SEHC": "Owners' Equivalent Rent",
        "CUSR0000SEHB": "Lodging Away From Home",
        "CUSR0000SEMF01": "Prescription Drugs",
        "CUSR0000SEMC01": "Physicians' Services",
        "CUSR0000SEMC02": "Dental Services",
        "CUSR0000SEMD01": "Hospital Services",
        "CUSR0000SETD": "Motor Vehicle Maintenance & Repair",
        "CUSR0000SETE": "Motor Vehicle Insurance",
        "CUSR0000SETG01": "Airline Fares"
    }

    payload = {
        "seriesid": list(ids.keys()),
        "startyear": str(CURRENT_YEAR - 2),
        "endyear": str(CURRENT_YEAR),
        "registrationkey": API_KEY
    }

    data = fetch_bls(payload)

    if data is None:
        return

    dfs = []

    for s in data["Results"]["series"]:
        sid = s["seriesID"]

        if sid not in ids:
            continue

        name = ids[sid]

        df = pd.DataFrame(s["data"])

        # Keep monthly observations only
        df = df[df["period"].str.startswith("M")].copy()

        df["Date"] = pd.to_datetime(
            df["year"] + "-" + df["period"].str.replace("M", "", regex=False),
            format="%Y-%m",
            errors="coerce"
        )

        df["Value"] = pd.to_numeric(
            df["value"],
            errors="coerce"
        )

        df = df.sort_values("Date")

        # Calculate m/m percentage change
        df[name] = (
            df["Value"] / df["Value"].shift(1) - 1
        ) * 100

        dfs.append(
            df[["Date", name]]
        )

    # Stop if BLS returned no usable series
    if not dfs:
        st.info("No CPI → PCE component data available.")
        return

    # --------------------------------------------------
    # Merge all component series
    # --------------------------------------------------
    final = dfs[0]

    for extra in dfs[1:]:
        final = final.merge(
            extra,
            on="Date",
            how="outer"
        )

    final = (
        final
        .sort_values("Date", ascending=False)
        .set_index("Date")
        .round(2)
    )

    # Format date
    final.index = final.index.strftime("%Y-%m")

    # --------------------------------------------------
    # Display table
    # --------------------------------------------------
    display_df = final.head(24)

    st.subheader("CPI → PCE Components (m/m %)")
    st.dataframe(
        display_df,
        use_container_width=True
    )

    # --------------------------------------------------
    # Headline generator
    # --------------------------------------------------
    if not display_df.empty:

        latest = display_df.iloc[0]

        if len(display_df) > 1:
            prev = display_df.iloc[1]
        else:
            prev = None

        headline_lines = []

        for col in display_df.columns:

            latest_val = latest[col]

            if prev is not None:
                prev_val = prev[col]
            else:
                prev_val = pd.NA

            # Skip component if latest value is unavailable
            if pd.isna(latest_val):
                continue

            if not pd.isna(prev_val):
                headline_lines.append(
                    f"{col}: {latest_val:.2f}% "
                    f"(prev. {prev_val:.2f}%)"
                )
            else:
                headline_lines.append(
                    f"{col}: {latest_val:.2f}% "
                    f"(prev. N/A)"
                )

        headline_text = "\n".join(headline_lines)

        st.markdown("### CPI → PCE Components headline format")

        st.text_area(
            "",
            value=headline_text,
            height=300,
            key="cpi_pce_headline"
        )

    # --------------------------------------------------
    # JOLTS headline generator
    # --------------------------------------------------
    if len(df) >= 2:
        latest = df.iloc[0]
        prev = df.iloc[1]

        headline_lines = []

        for col in df.columns:
            latest_val = latest[col]
            prev_val = prev[col]

            if "rate" in col.lower():
                current_text = "N/A" if pd.isna(latest_val) else f"{latest_val:.1f}%"
                prev_text = "N/A" if pd.isna(prev_val) else f"{prev_val:.1f}%"
            else:
                current_text = "N/A" if pd.isna(latest_val) else f"{latest_val:,.0f}k"
                prev_text = "N/A" if pd.isna(prev_val) else f"{prev_val:,.0f}k"

            headline_lines.append(
                f"{col}: {current_text} (prev. {prev_text})"
            )

        st.markdown("### JOLTS headline format")
        st.text_area(
            "",
            value="\n".join(headline_lines),
            height=320,
            key="jolts_headline"
        )

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
        "CPI → PCE Components",
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
    elif choice == "CPI → PCE Components":
    run_cpi_pce()    
    elif choice == "PPI → PCE Components":
        run_ppi_pce()    
    elif choice == "JOLTS":
        run_jolts()
    elif choice == "NFP & Unemployment":
        run_nfp()
