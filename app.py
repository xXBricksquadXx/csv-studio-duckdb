# app.py — CSV Studio (Minimal, Premium Noir+Gold, Gremlin-free)
import io
import duckdb
import pandas as pd
import plotly.express as px
import streamlit as st

st.set_page_config(page_title="CSV Studio — DuckDB", page_icon="🦆", layout="wide")

# -----------------------------
# Themes (Noir + Gold and Light)
# -----------------------------
GOLD_DARK_CSS = """
<style>
:root{
  /* Premium Noir + Gold */
  --bg:#0A0C10;             /* deep charcoal */
  --surface:#10151C;        /* card bg */
  --surface-2:#0C1117;      /* sidebar / controls */
  --border:#1B2430;         /* soft steel */
  --text:#EAF0F7;           /* crisp text */
  --muted:#9FB0C1;          /* secondary */
  --accent:#E5C558;         /* warm gold */
  --accent-600:#C9A43E;
  --accent-700:#9A7E2B;
  --shadow:0 14px 40px rgba(0,0,0,.45);
}

/* Background: richer black→gold blend */
[data-testid="stAppViewContainer"]{
  background:
    radial-gradient(1200px 600px at 14% -10%, rgba(229,197,88,.18), rgba(0,0,0,0) 48%),
    radial-gradient(1000px 520px at 102% 0%, rgba(229,197,88,.13), rgba(0,0,0,0) 42%),
    linear-gradient(180deg, #0A0C10 0%, #0B0F14 60%, #090C10 100%);
}

/* Sidebar */
section[data-testid="stSidebar"]{
  background:var(--surface-2);
  border-right:1px solid var(--border);
}

/* Typography */
h1,h2,h3,h4,h5,h6{ color:var(--text)!important; }
p,label,span,code,.stMarkdown{ color:var(--muted); }

/* Cards / blocks */
.block-container>div,
div[data-testid="stVerticalBlock"]>div,
div[data-testid="stHorizontalBlock"]>div{ background:transparent; }

div[data-testid="stVerticalBlockBorderWrapper"]{
  background:
    linear-gradient(180deg, rgba(255,255,255,.02), rgba(255,255,255,0)) padding-box,
    radial-gradient(600px 240px at 10% -20%, rgba(229,197,88,.35), rgba(229,197,88,.06)) border-box;
  border:1px solid rgba(229,197,88,.28);
  border-radius:16px;
  box-shadow:var(--shadow);
}

/* Inputs / selects */
.stTextInput input,.stNumberInput input,.stDateInput input,
.stSelectbox div[data-baseweb="select"]>div{
  background:var(--surface);
  color:var(--text);
  border:1px solid var(--border);
  border-radius:10px;
}
div[data-baseweb="popover"] div[role="listbox"]{
  background:var(--surface);
  color:var(--text);
  border:1px solid var(--border);
}
section[data-testid="stSidebar"] input::placeholder{ color:#A9B8C8!important; opacity:.9; }

/* Sidebar file uploader */
section[data-testid="stSidebar"] [data-testid="stFileUploader"]{
  background:var(--surface);
  border:1px solid var(--border);
  border-radius:14px;
  box-shadow:var(--shadow);
  padding:10px 12px;
}
section[data-testid="stSidebar"] [data-testid="stFileUploader"] *{ color:var(--text)!important; }
section[data-testid="stSidebar"] [data-testid="stFileUploaderDropzone"]{
  background:var(--surface);
  border:1px dashed var(--border);
}
section[data-testid="stSidebar"] [data-testid="stFileUploader"] button{
  background:var(--accent)!important; border:1px solid var(--accent-600)!important; color:#111!important;
}
section[data-testid="stSidebar"] [data-testid="stFileUploader"] button *{
  color:#111!important; text-shadow:none!important; mix-blend-mode:normal!important;
}

/* Buttons (with inner text/icon fix) */
.stButton>button, .stDownloadButton>button{
  background:var(--accent)!important;
  border:1px solid var(--accent-600)!important;
  color:#111!important;
  border-radius:12px!important;
  padding:8px 14px!important;
  box-shadow:0 6px 18px rgba(229,197,88,.28)!important;
  text-shadow:none!important;
}
.stButton>button * , .stDownloadButton>button * { color:#111!important; text-shadow:none!important; }
.stButton>button:hover, .stDownloadButton>button:hover{ background:var(--accent-600)!important; border-color:var(--accent-700)!important; }
.stButton>button:disabled, .stDownloadButton>button:disabled{
  background:var(--accent)!important; border-color:var(--accent-600)!important; color:#111!important;
  opacity:.65!important; box-shadow:none!important;
}

/* Tabs: subtle glass + gold underline */
div[data-baseweb="tab-list"]{
  background:rgba(16,21,28,.7);
  backdrop-filter:blur(8px);
  border:1px solid var(--border);
  border-radius:12px;
  box-shadow:var(--shadow);
}
button[role="tab"]{ color:#A9B4C0; }
button[role="tab"][aria-selected="true"]{
  color:var(--text);
  border-bottom:2px solid var(--accent);
}

/* Table / DataFrame */
.stDataFrame, .stDataFrame [data-testid="stTable"]{
  background:var(--surface);
  color:var(--text);
  border:1px solid var(--border);
  border-radius:12px;
  box-shadow:var(--shadow);
}
.stDataFrame thead th{
  background:rgba(229,197,88,.10);
  color:var(--text);
  border-bottom:1px solid var(--border);
}
/* sticky table header */
.stDataFrame [data-testid="stTable"] thead tr th{
  position: sticky; top: 0; z-index: 3; background: var(--surface);
}

/* KPI metrics: brighter values */
[data-testid="stMetricLabel"]{ color:var(--muted)!important; }
[data-testid="stMetricValue"]{ color:var(--text)!important; }

/* Chips / multiselect tokens */
div[role="listitem"]{
  background:rgba(229,197,88,.14)!important;
  color:#F3E9C2!important;
  border:1px solid rgba(229,197,88,.35)!important;
  border-radius:12px!important;
}

/* Slider handle in gold */
.stSlider [role="slider"]{ background:var(--accent)!important; }

/* Welcome card */
.hero-card{
  background:
    linear-gradient(180deg, rgba(255,255,255,.02), rgba(255,255,255,0)) padding-box,
    radial-gradient(1000px 400px at 14% -10%, rgba(229,197,88,.22), rgba(229,197,88,.05)) border-box;
  border:1px solid rgba(229,197,88,.26);
  border-radius:18px!important;
  box-shadow:var(--shadow)!important;
  padding:32px!important;
}
</style>
"""

LIGHT_CSS = """
<style>
:root{
  --bg:#FAFAF9; --surface:#FFFFFF; --border:#E7E5E4; --text:#0F172A; --muted:#475569;
  --accent:#D4AF37; --accent-600:#B99024; --accent-700:#946E12; --shadow:0 6px 18px rgba(15,23,42,0.05);
}
[data-testid="stAppViewContainer"]{ background:var(--bg); }
section[data-testid="stSidebar"]{ background:var(--surface); border-right:1px solid var(--border); }
h1,h2,h3,h4,h5,h6{ color:var(--text)!important; } p,label,span,code,.stMarkdown{ color:var(--text); }
div[data-testid="stVerticalBlockBorderWrapper"]{ background:var(--surface); border:1px solid var(--border); border-radius:16px; box-shadow:var(--shadow); }
.stTextInput input,.stNumberInput input,.stDateInput input,
.stSelectbox div[data-baseweb="select"]>div{ background:var(--surface); color:var(--text); border:1px solid var(--border); border-radius:10px; }
div[data-baseweb="popover"] div[role="listbox"]{ background:var(--surface); color:var(--text); border:1px solid var(--border); }
section[data-testid="stSidebar"] input::placeholder{ color:#6B7280!important; opacity:.9; }
section[data-testid="stSidebar"] [data-testid="stFileUploader"]{ background:var(--surface); border:1px solid var(--border); border-radius:14px; box-shadow:var(--shadow); padding:10px 12px; }
section[data-testid="stSidebar"] [data-testid="stFileUploader"] button{
  background:var(--accent)!important; border:1px solid var(--accent-600)!important; color:#111!important;
}
/* keep inner spans black in light mode too */
section[data-testid="stSidebar"] [data-testid="stFileUploader"] button * ,
.stButton>button * , .stDownloadButton>button * { color:#111!important; }
.stButton>button, .stDownloadButton>button{
  background:var(--accent)!important; border:1px solid var(--accent-600)!important; color:#111!important;
  border-radius:12px!important; padding:8px 14px!important; box-shadow:0 4px 10px rgba(212,175,55,0.15)!important;
}
.stButton>button:hover, .stDownloadButton>button:hover{ background:var(--accent-600)!important; border-color:var(--accent-700)!important; }
.stButton>button:disabled, .stDownloadButton>button:disabled{
  background:var(--accent)!important; border-color:var(--accent-600)!important; color:#111!important; opacity:.65!important; box-shadow:none!important;
}
div[data-baseweb="tab-list"]{ background:var(--surface); border:1px solid var(--border); border-radius:12px; box-shadow:var(--shadow); }
button[role="tab"]{ color:#64748B; } button[role="tab"][aria-selected="true"]{ color:var(--text); border-bottom:2px solid var(--accent); }
.stDataFrame, .stDataFrame [data-testid="stTable"]{ background:var(--surface); color:var(--text); border:1px solid var(--border); border-radius:12px; box-shadow:var(--shadow); }
.stDataFrame thead th{ background:#FFFDF4; color:var(--text); border-bottom:1px solid var(--border); }
/* sticky table header */
.stDataFrame [data-testid="stTable"] thead tr th{
  position: sticky; top: 0; z-index: 3; background: var(--surface);
}
.hero-card{ background:var(--surface)!important; border:1px solid var(--border)!important; border-radius:18px!important; box-shadow:var(--shadow)!important; padding:32px!important; }
</style>
"""

# -----------------------------
# Helper functions
# -----------------------------
def ident(name: str) -> str:
    s = str(name)
    try:
        if hasattr(duckdb, "escape_identifier"):
            return duckdb.escape_identifier(s)
    except Exception:
        pass
    return '"' + s.replace('"', '""') + '"'

def ensure_id(df: pd.DataFrame) -> pd.DataFrame:
    if "_id" not in df.columns:
        df = df.copy()
        df["_id"] = range(1, len(df) + 1)
    return df

def coerce_dates(df: pd.DataFrame) -> pd.DataFrame:
    for c in df.columns:
        if "date" in c.lower():
            try: df[c] = pd.to_datetime(df[c], errors="ignore")
            except Exception: pass
    return df

def read_any(file_or_url, filename_hint: str | None = None) -> pd.DataFrame:
    name = (getattr(file_or_url, "name", None) or filename_hint or "").lower()
    if name.endswith((".xlsx", ".xls")): return pd.read_excel(file_or_url)
    if name.endswith(".tsv") or ("tsv" in name and not name.endswith(".csv")):
        return pd.read_csv(file_or_url, sep="\t")
    return pd.read_csv(file_or_url)

def sample_df() -> pd.DataFrame:
    d = pd.date_range("2025-08-01", periods=6, freq="D")
    df = pd.DataFrame({
        "date": d,
        "region": ["East","West","East","West","East","West"],
        "product": ["Widget A","Widget B","Widget A","Widget B","Widget A","Widget B"],
        "orders": [23,15,21,10,12,16],
        "revenue": [1840,1200,1680,800,960,1280],
    })
    return ensure_id(df)

def to_duck(df: pd.DataFrame):
    con = duckdb.connect(database=":memory:")
    con.register("df", df)
    return con

def paged(df: pd.DataFrame, page: int, per_page: int):
    total_pages = max(1, (len(df) - 1) // per_page + 1)
    page = max(1, min(page, total_pages))
    start, end = (page - 1) * per_page, (page - 1) * per_page + per_page
    return df.iloc[start:end].copy(), total_pages, page

def reset_filters_for(df: pd.DataFrame):
    cols = [c for c in df.columns if c != "_id"]
    cat_col = cols[0] if cols else None
    num_cols = [c for c in cols if pd.api.types.is_numeric_dtype(df[c]) or pd.api.types.is_bool_dtype(df[c])]
    metric = next((c for c in num_cols if c != cat_col), (num_cols[0] if num_cols else None))
    f = {"q":"", "cat_col":cat_col, "cat_vals":[], "metric":metric, "metric_range":None, "date_col":"(none)"}
    if metric is not None:
        s = df[metric]
        rng_series = s.astype(int) if s.dtype == bool else s
        f["metric_range"] = (float(rng_series.min()), float(rng_series.max()))
    st.session_state.filters = f
    st.session_state.page = 1

# make Plotly backgrounds transparent so CSS shines through
def _make_fig_bg_transparent(fig):
    fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
    return fig

# share view helper (URL params)
def _put_query_params(params: dict):
    try:
        st.query_params.clear(); st.query_params.update(params)
    except Exception:
        st.experimental_set_query_params(**params)

# -----------------------------
# Session state
# -----------------------------
ss = st.session_state
ss.setdefault("df", None); ss.setdefault("filters", {}); ss.setdefault("page", 1)
ss.setdefault("rows_per_page", 50); ss.setdefault("light_theme", False)
ss.setdefault("loaded_token", None); ss.setdefault("uploader_key", 0)

# -----------------------------
# Sidebar
# -----------------------------
with st.sidebar:
    st.markdown("### Appearance")
    ss.light_theme = st.toggle("Light theme", value=ss.light_theme)
    st.markdown(LIGHT_CSS if ss.light_theme else GOLD_DARK_CSS, unsafe_allow_html=True)
    px.defaults.template = "plotly_white" if ss.light_theme else "plotly_dark"

    st.divider()
    st.markdown("### Load data")
    uploaded = st.file_uploader("CSV / TSV / XLSX", type=["csv", "tsv", "xlsx"], key=f"uploader_{ss.uploader_key}")
    url = st.text_input("…or CSV URL", placeholder="https://example.com/data.csv")
    c1, _, c3 = st.columns([1,1,1])
    load_sample = c1.button("Load sample")
    fetch = c3.button("Fetch")
    clear = st.button("Clear data")

    if clear:
        ss.df = None; ss.loaded_token = None; ss.page = 1
        ss.uploader_key += 1
        st.toast("Dataset cleared."); st.rerun()

    if load_sample:
        ss.df = coerce_dates(sample_df()); ss.loaded_token = "sample"
        reset_filters_for(ss.df); ss.uploader_key += 1
        st.toast("Loaded sample dataset."); st.rerun()

    elif uploaded is not None:
        token = f"{uploaded.name}:{getattr(uploaded, 'size', 0)}"
        if ss.loaded_token != token:
            ss.df = coerce_dates(ensure_id(read_any(uploaded, uploaded.name)))
            ss.loaded_token = token; reset_filters_for(ss.df)
            st.success(f"Loaded: {uploaded.name}")

    elif fetch and url.strip():
        try:
            ss.df = coerce_dates(ensure_id(read_any(url, url)))
            ss.loaded_token = f"url:{url}"
            reset_filters_for(ss.df); ss.uploader_key += 1
            st.success("Loaded from URL."); st.rerun()
        except Exception as e:
            st.error(f"Failed to fetch: {e}")

    st.divider()
    st.markdown("### Filters")
    f = ss.filters
    f["q"] = st.text_input("Global search", value=f.get("q", ""))

    if st.button("Clear filters"):
        reset_filters_for(ss.df); st.rerun()

    if ss.df is not None:
        cols = [c for c in ss.df.columns if c != "_id"]

        # Category
        cat_default = f.get("cat_col", cols[0] if cols else None)
        cat_idx = cols.index(cat_default) if (cat_default in cols) else (0 if cols else 0)
        f["cat_col"] = st.selectbox("Category column", options=cols, index=cat_idx)
        cat_vals_all = sorted(ss.df[f["cat_col"]].dropna().astype(str).unique().tolist())[:500]
        f["cat_vals"] = st.multiselect("Category values", options=cat_vals_all,
                                       default=[v for v in f.get("cat_vals", []) if v in cat_vals_all])

        # Metric (numeric or boolean) — robust slider for legacy/edge cases
        num_cols = [c for c in cols if pd.api.types.is_numeric_dtype(ss.df[c]) or pd.api.types.is_bool_dtype(ss.df[c])]
        if num_cols:
            preferred = next((c for c in num_cols if c != f["cat_col"]), num_cols[0])
            cur_metric = f.get("metric", preferred)
            if cur_metric not in num_cols:
                cur_metric = preferred
            f["metric"] = st.selectbox("Metric (KPIs & charts)", options=num_cols, index=num_cols.index(cur_metric))

            s = ss.df[f["metric"]]
            s_clean = s.dropna()

            if s_clean.empty:
                st.info("No numeric values in this column.")
                f["metric_range"] = None

            elif pd.api.types.is_bool_dtype(s):
                lo_i, hi_i = int(s_clean.min()), int(s_clean.max())
                if lo_i == hi_i:
                    st.caption("Metric is constant (all the same).")
                    f["metric_range"] = (float(lo_i), float(hi_i))
                else:
                    cur_lo, cur_hi = f.get("metric_range", (lo_i, hi_i))
                    cur_lo = int(max(lo_i, min(int(cur_lo), hi_i)))
                    cur_hi = int(max(lo_i, min(int(cur_hi), hi_i)))
                    f["metric_range"] = st.slider("Metric range", min_value=lo_i, max_value=hi_i,
                                                  value=(cur_lo, cur_hi), step=1)

            elif pd.api.types.is_integer_dtype(s):
                lo_i, hi_i = int(s_clean.min()), int(s_clean.max())
                if lo_i == hi_i:
                    st.caption("Metric is constant (all the same).")
                    f["metric_range"] = (float(lo_i), float(hi_i))
                else:
                    cur_lo, cur_hi = f.get("metric_range", (lo_i, hi_i))
                    cur_lo = int(max(lo_i, min(int(cur_lo), hi_i)))
                    cur_hi = int(max(lo_i, min(int(cur_hi), hi_i)))
                    f["metric_range"] = st.slider("Metric range", min_value=lo_i, max_value=hi_i,
                                                  value=(cur_lo, cur_hi), step=1)

            else:
                lo_f, hi_f = float(s_clean.min()), float(s_clean.max())
                if not (hi_f > lo_f):
                    st.caption("Metric has no varying values.")
                    f["metric_range"] = (lo_f, hi_f)
                else:
                    cur_lo, cur_hi = f.get("metric_range", (lo_f, hi_f))
                    cur_lo = float(max(lo_f, min(float(cur_lo), hi_f)))
                    cur_hi = float(max(lo_f, min(float(cur_hi), hi_f)))
                    f["metric_range"] = st.slider("Metric range", min_value=lo_f, max_value=hi_f,
                                                  value=(cur_lo, cur_hi))
        else:
            f["metric"] = None
            f["metric_range"] = None

        # Date (optional)
        date_opts = ["(none)"] + [c for c in cols if "date" in c.lower() or pd.api.types.is_datetime64_any_dtype(ss.df[c])]
        cur_date = f.get("date_col", "(none)")
        if cur_date not in date_opts: cur_date = "(none)"
        f["date_col"] = st.selectbox("Date column (for time series)", options=date_opts,
                                     index=date_opts.index(cur_date))

        ss.rows_per_page = st.selectbox("Rows per page", options=[25, 50, 100, 250, 1000],
                                        index=[25, 50, 100, 250, 1000].index(ss.rows_per_page))

    # Share current view (URL params)
    if st.button("🔗 Share this view"):
        _put_query_params({
            "q": f.get("q",""),
            "cat_col": f.get("cat_col",""),
            "cat_vals": ",".join(f.get("cat_vals",[])),
            "metric": f.get("metric",""),
            "mr": ",".join([str(x) for x in (f.get("metric_range") or [])]),
            "date_col": f.get("date_col",""),
            "rpp": str(ss.rows_per_page),
            "p": str(ss.page),
            "light": "1" if ss.light_theme else "0",
        })
        st.toast("URL updated — copy from the address bar.")

# -----------------------------
# Title / Empty state
# -----------------------------
st.title("CSV Studio — DuckDB Edition")
st.caption("Load → filter → KPIs & charts → edit → export")

df = ss.df
if df is None or len(df) == 0:
    st.markdown("""
    <div class="hero-card">
      <h2 style="margin:0 0 8px 0;">Welcome</h2>
      <p style="margin:0;">Load a dataset from the sidebar to get started.</p>
    </div>
    """, unsafe_allow_html=True)
    st.stop()

# -----------------------------
# Build filtered view (DuckDB)
# -----------------------------
con = to_duck(df)
q = "SELECT * FROM df"
clauses, params = [], []

qtext = f.get("q", "").strip()
if qtext:
    text_cols = [c for c in df.columns if df[c].dtype == "object" or df[c].dtype.name.startswith("string")]
    if text_cols:
        ors = []
        for c in text_cols:
            ors.append(f"LOWER(CAST({ident(c)} AS VARCHAR)) LIKE LOWER(?)")
            params.append(f"%{qtext}%")
        clauses.append("(" + " OR ".join(ors) + ")")

cat_col = f.get("cat_col"); cat_vals = f.get("cat_vals", [])
if cat_col and cat_vals:
    placeholders = ",".join(["?"] * len(cat_vals))
    clauses.append(f"{ident(cat_col)} IN ({placeholders})")
    params.extend(cat_vals)

metric = f.get("metric"); mr = f.get("metric_range")
metric_is_bool = bool(metric) and pd.api.types.is_bool_dtype(df[metric]) if metric else False
metric_expr = f"(CASE WHEN {ident(metric)} THEN 1 ELSE 0 END)" if metric_is_bool else ident(metric)

if metric and mr:
    lo, hi = mr
    clauses.append(f"{metric_expr} BETWEEN ? AND ?")
    params.extend([float(lo), float(hi)])

if clauses:
    q += " WHERE " + " AND ".join(clauses)

if metric:
    q += f" ORDER BY {metric_expr} DESC"

try:
    view = con.execute(q, params).df()
except Exception as e:
    st.warning(f"Filter query failed: {e}")
    view = df.copy()

# -----------------------------
# KPIs
# -----------------------------
k1, k2, k3 = st.columns(3)
with k1: st.metric("Rows", len(view))
with k2:
    if metric and metric in view:
        s = view[metric].astype(int) if metric_is_bool else view[metric]
        if pd.api.types.is_numeric_dtype(s): st.metric(f"Sum {metric}", f"{s.sum():,.2f}")
with k3:
    if metric and metric in view:
        s = view[metric].astype(int) if metric_is_bool else view[metric]
        if pd.api.types.is_numeric_dtype(s): st.metric(f"Median {metric}", f"{s.median():,.2f}")

# -----------------------------
# Tabs
# -----------------------------
tab_charts, tab_table = st.tabs(["Charts", "Table / Edit"])

with tab_charts:
    mode = st.radio("Chart", options=["Time series", "By category"], horizontal=True, index=1)
    date_col = f.get("date_col", "(none)")
    if mode == "Time series":
        if date_col != "(none)" and metric:
            s = view[metric].astype(int) if metric_is_bool else view[metric]
            if pd.api.types.is_numeric_dtype(s):
                tmp = view[[date_col]].join(s).dropna()
                if not tmp.empty:
                    fig = px.line(tmp, x=date_col, y=metric, title=f"{metric} over time")
                    fig = _make_fig_bg_transparent(fig)
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.info("No data for selected date/metric.")
            else:
                st.info("Choose a numeric/boolean metric.")
        else:
            st.info("Choose a date column and metric.")
    else:
        if cat_col and metric:
            s = view[metric].astype(int) if metric_is_bool else view[metric]
            if pd.api.types.is_numeric_dtype(s):
                grp = view.groupby(cat_col, dropna=False)[metric].sum().reset_index(name=f"{metric}_sum")
                grp = grp.sort_values(f"{metric}_sum", ascending=False).head(40)
                fig = px.bar(grp, x=cat_col, y=f"{metric}_sum", title=f"{metric} by {cat_col}")
                fig = _make_fig_bg_transparent(fig)
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("Pick a numeric/boolean metric to see a bar chart.")
        else:
            st.info("Pick a category + metric to see a bar chart.")

with tab_table:
    # Quick filter (updates global search)
    q_inline = st.text_input("Quick filter (any column)", value=f.get("q",""),
                             placeholder="Type to filter rows…", key="q_inline")
    if q_inline != f.get("q",""):
        f["q"] = q_inline
        st.rerun()

    # Pagination
    per_page = ss.rows_per_page
    page_df, total_pages, clamped = paged(view, ss.page, per_page)
    if clamped != ss.page: ss.page = clamped

    # Editor
    disp_cols = [c for c in view.columns if c != "_id"]
    page_df = page_df[disp_cols + (["_id"] if "_id" in view.columns else [])].copy()
    if "Delete" not in page_df.columns: page_df["Delete"] = False

    edited = st.data_editor(page_df, hide_index=True, use_container_width=True, num_rows="dynamic",
                            key=f"editor_{ss.page}")

    cprev, cpage, cnext = st.columns([1,2,1])
    with cprev:
        if st.button("◀ Prev", key=f"prev_{ss.page}", use_container_width=True, disabled=ss.page <= 1):
            ss.page -= 1; st.rerun()
    with cpage:
        st.markdown(f"<div style='text-align:center'>Page <b>{ss.page}</b> / {total_pages}</div>", unsafe_allow_html=True)
    with cnext:
        if st.button("Next ►", key=f"next_{ss.page}", use_container_width=True, disabled=ss.page >= total_pages):
            ss.page += 1; st.rerun()

    ca, cb, cc = st.columns([1,1,1])
    with ca:
        if st.button("Apply changes", use_container_width=True, key=f"apply_{ss.page}"):
            base = df.set_index("_id"); ed = edited.set_index("_id")
            to_drop = ed.index[ed.get("Delete", False) == True].tolist()  # noqa: E712
            base = base.drop(index=[i for i in to_drop if i in base.index])
            if "Delete" in ed.columns: ed = ed.drop(columns=["Delete"])
            ed = ed.reindex(columns=base.columns, fill_value=pd.NA)
            for _id, row in ed.iterrows(): base.loc[_id, :] = row
            ss.df = base.reset_index().sort_values("_id")
            st.success("Changes applied."); st.rerun()

    with cb:
        st.download_button(
            "Download page CSV",
            data=edited.to_csv(index=False).encode(),
            file_name="page.csv",
            mime="text/csv",
            use_container_width=True,
            key=f"dlpage_{ss.page}",
        )
    with cc:
        buf = io.BytesIO(); ss.df.to_parquet(buf, index=False)
        st.download_button(
            "⬇️ Full dataset (Parquet)",
            data=buf.getvalue(),
            file_name="dataset.parquet",
            mime="application/octet-stream",
            use_container_width=True,
            key=f"dlfull_{ss.page}",
        )
