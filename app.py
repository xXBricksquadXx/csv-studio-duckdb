import math
from typing import Dict, Any

import duckdb
import pandas as pd
import plotly.express as px
import streamlit as st

# ============================== Page setup ==============================
st.set_page_config(page_title="CSV Studio — DuckDB Edition", layout="wide")

# Clean screenshot mode: add ?shot=1 to the URL to hide UI chrome
qs = st.query_params
if "shot" in qs:
    st.markdown("""
    <style>
      [data-testid="stSidebar"]{display:none;}
      header, footer, #MainMenu{visibility:hidden;}
    </style>
    """, unsafe_allow_html=True)

# Saints black & gold + glossy UI
st.markdown("""
<style>
:root{
  --black:#0B0B0B; --panel:#121214;
  --gold:#D2B887; --fg:#FFF8E7;
}
html, body, [data-testid="stAppViewContainer"]{
  background:
    radial-gradient(1000px 500px at 20% -10%, rgba(210,184,135,.08), transparent 60%),
    radial-gradient(900px 500px at 120% 10%, rgba(210,184,135,.06), transparent 55%),
    var(--black);
  color:var(--fg);
}
.block-container{padding-top:1rem}
[data-testid="stSidebar"]{
  background:linear-gradient(180deg,#0D0D0D,#101213);
  border-right:1px solid rgba(210,184,135,.20);
}
h1,h2,h3{letter-spacing:.3px}

/* Buttons */
.stButton>button{
  background:linear-gradient(180deg,#1a1a1a,#0f0f0f);
  color:var(--fg);
  border:1px solid rgba(210,184,135,.55);
  border-radius:14px; padding:.55rem .9rem;
  box-shadow: inset 0 1px 0 rgba(255,255,255,.06),
              0 0 0 2px rgba(210,184,135,.1),
              0 12px 20px rgba(0,0,0,.35);
  transition:transform .15s ease, box-shadow .2s ease, border-color .2s;
}
.stButton>button:hover{
  transform:translateY(-1px);
  border-color:var(--gold);
  box-shadow:0 0 0 3px rgba(210,184,135,.20),0 14px 22px rgba(0,0,0,.45);
}

/* Tabs */
.stTabs [data-baseweb="tab-list"]{gap:.5rem}
.stTabs [data-baseweb="tab"]{
  border:1px solid rgba(210,184,135,.25);
  border-bottom-color:rgba(210,184,135,.35);
  background:linear-gradient(180deg,#111113,#0d0d0e);
  color:var(--fg); border-radius:12px 12px 0 0;
}
.stTabs [aria-selected="true"]{
  background:radial-gradient(120% 120% at 50% -30%, rgba(210,184,135,.25), transparent 60%), #111113;
  border-color:rgba(210,184,135,.6);
}

/* “Oozy” hero when empty */
.hero{
  position:relative; border-radius:18px; padding:3.5rem 2rem; text-align:center;
  border:1px solid rgba(210,184,135,.4);
  background:radial-gradient(600px 220px at 50% -10%, rgba(210,184,135,.18), transparent 60%), #0b0b0b;
  overflow:hidden;
}
.blob, .blob2{
  position:absolute; filter:blur(26px); opacity:.25; mix-blend-mode:screen;
  background:
    radial-gradient(circle at 30% 30%, rgba(210,184,135,.9), transparent 60%),
    radial-gradient(circle at 70% 70%, rgba(255,223,128,.6), transparent 60%);
  width:320px; height:320px; border-radius:50%;
  animation:ooze 12s ease-in-out infinite;
}
.blob{left:-60px; top:-30px}
.blob2{right:-60px; top:20px; animation-duration:14s}
@keyframes ooze{0%{transform:translate(-40px,-10px) scale(1)}50%{transform:translate(20px,10px) scale(1.15)}100%{transform:translate(-40px,-10px) scale(1)}}
.fade-in{animation:fade .25s ease-out}
@keyframes fade{from{opacity:0; transform:translateY(4px)} to{opacity:1; transform:none}}
/* Compact icon buttons in sidebar */
[data-testid="stSidebar"] .stButton>button{padding:.4rem .5rem; border-radius:10px}
</style>
""", unsafe_allow_html=True)

st.title("CSV Studio — DuckDB Edition")
st.caption("Upload CSV/TSV/XLSX or load from URL → filter → KPIs & charts → edit (CRUD) → export.")

# ============================== Session defaults ==============================
defaults: Dict[str, Any] = {
    "df": None,
    "last_df": None,
    "next_id": 0,
    "last_url": None,
    "last_upload_sig": None,  # NEW: remember last uploaded file (name, size)
    "page": 1,
    "uploader_key": 0,
    "page_size": 25,
    "sort_col": "",
    "sort_dir": "Ascending",
    "filters_nonce": 0,       # force widget refresh on new dataset / clear
}
for k, v in defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v

# ============================== Helpers ==============================
def bump_filters_nonce():
    st.session_state.filters_nonce += 1

def clear_filters():
    for key in [
        "text_col","q","cat_col","cat_vals","date_col","date_rng",
        "metric_col","metric_range","q_all","needle","visible_cols"
    ]:
        if key in st.session_state:
            del st.session_state[key]
    st.session_state.sort_col = ""
    st.session_state.sort_dir = "Ascending"
    st.session_state.page_size = 25
    st.session_state.page = 1
    bump_filters_nonce()

def clear_data():
    st.session_state.df = None
    st.session_state.last_df = None
    st.session_state.next_id = 0
    st.session_state.last_url = None
    st.session_state.last_upload_sig = None  # reset uploaded-file memo
    st.session_state.uploader_key += 1
    st.session_state.use_sample = False
    st.session_state.csv_url = ""
    clear_filters()
    st.toast("Data cleared.", icon="🗑️")
    st.rerun()

def reset_app():
    st.session_state.clear()
    st.rerun()

def parse_dates_best_effort(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for c in out.columns:
        if out[c].dtype == object:
            try:
                parsed = pd.to_datetime(out[c], errors="raise")
                if parsed.notna().mean() > 0.5:
                    out[c] = parsed
            except Exception:
                pass
    return out

def ensure_id(df: pd.DataFrame) -> pd.DataFrame:
    df2 = df.copy()
    if "_id" not in df2.columns:
        start = st.session_state.next_id
        df2["_id"] = range(start, start + len(df2))
        st.session_state.next_id = start + len(df2)
    else:
        try:
            st.session_state.next_id = max(int(df2["_id"].max()) + 1, st.session_state.next_id)
        except Exception:
            start = st.session_state.next_id
            df2["_id"] = range(start, start + len(df2))
            st.session_state.next_id = start + len(df2)
    return df2

def register_duck(df: pd.DataFrame):
    con = duckdb.connect(database=":memory:")
    con.register("data", df)
    return con

def read_any(uploaded) -> pd.DataFrame:
    name = uploaded.name.lower()
    if name.endswith(".csv"):
        return pd.read_csv(uploaded)
    if name.endswith(".tsv"):
        return pd.read_csv(uploaded, sep="\t")
    if name.endswith(".xlsx"):
        return pd.read_excel(uploaded)  # needs openpyxl
    raise ValueError("Unsupported format (use .csv, .tsv, or .xlsx)")

def on_new_dataset_loaded(new_df: pd.DataFrame, source_msg: str):
    new_df = ensure_id(parse_dates_best_effort(new_df))
    st.session_state.df = new_df
    st.session_state.last_df = None
    st.session_state.page = 1
    st.session_state.sort_col = ""
    st.session_state.sort_dir = "Ascending"
    clear_filters()  # also bumps nonce
    st.toast(source_msg, icon="✅")

# ============================== Sidebar controls ==============================
with st.sidebar:
    st.markdown("### Data controls")
    c1, c2, c3 = st.columns(3)
    if c1.button("🧹", help="Clear filters", use_container_width=True):
        clear_filters(); st.rerun()
    if c2.button("🗑️", help="Clear data", use_container_width=True):
        clear_data()
    if c3.button("🔄", help="Reset app", use_container_width=True):
        reset_app()

# ============================== 1) Load data ==============================
st.sidebar.markdown("### 1) Load data")
uploaded = st.sidebar.file_uploader(
    "Upload a file", type=["csv","tsv","xlsx"],
    key=f"uploader_{st.session_state.uploader_key}",
)
use_sample = st.sidebar.checkbox("Use sample/toy.csv",
                                 value=st.session_state.get("use_sample", False),
                                 key="use_sample")

st.sidebar.divider()
st.sidebar.subheader("Or load from URL (CSV)")
csv_url = st.sidebar.text_input("CSV URL", key="csv_url", placeholder="https://...")
fetch = st.sidebar.button("Fetch URL", key="fetch_url")

# --- File upload: load only when the actual file changes
if uploaded is not None:
    sig = (uploaded.name, getattr(uploaded, "size", None))
    if st.session_state.df is None or st.session_state.last_upload_sig != sig:
        try:
            base = read_any(uploaded)
            on_new_dataset_loaded(base, f"Loaded file: {uploaded.name}")
            st.session_state.last_upload_sig = sig
        except Exception as e:
            st.error(f"Could not read file: {e}")

    # Optional manual reload
    if st.sidebar.button("Reload uploaded file"):
        try:
            base = read_any(uploaded)
            on_new_dataset_loaded(base, f"Reloaded: {uploaded.name}")
            st.session_state.last_upload_sig = (uploaded.name, getattr(uploaded, "size", None))
        except Exception as e:
            st.error(f"Could not read file: {e}")

# --- URL fetch (only when clicked or URL changed)
elif (csv_url and (fetch or st.session_state.last_url != csv_url)):
    try:
        web_df = pd.read_csv(csv_url)
        if "time" in web_df.columns and pd.api.types.is_numeric_dtype(web_df["time"]):
            web_df["time"] = pd.to_datetime(web_df["time"], unit="ms")
        st.session_state.last_url = csv_url
        on_new_dataset_loaded(web_df, "Loaded data from URL.")
    except Exception as e:
        st.error(f"Could not fetch CSV: {e}")

# --- Sample (first load only)
elif st.session_state.df is None and use_sample:
    base = pd.read_csv("sample/toy.csv")
    on_new_dataset_loaded(base, "Loaded sample/toy.csv")

df = st.session_state.df
if df is None:
    st.markdown(
        """
        <div class="hero fade-in">
          <div class="blob"></div><div class="blob2"></div>
          <h1 style="margin:0 0 .25rem 0;">Welcome</h1>
          <p style="opacity:.9; margin-top:.25rem">
            Load a CSV/TSV/XLSX or paste a CSV URL to get started.<br/>
            Use the left toolbar to <b>clear filters</b>, <b>clear data</b>, or <b>reset</b>.
          </p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.stop()

# ============================== Types ==============================
num_cols = df.select_dtypes(include="number").columns.drop("_id", errors="ignore").tolist()
date_cols = df.select_dtypes(include="datetime64[ns]").columns.tolist()
txt_cols  = [c for c in df.columns if c not in num_cols + date_cols + ["_id"]]

# ============================== 2) Filters ==============================
nonce = st.session_state.filters_nonce
st.sidebar.markdown("### 2) Filters")

text_col = st.sidebar.selectbox("Search column", [""] + txt_cols, key=f"text_col_{nonce}")
q        = st.sidebar.text_input("Search text", "", key=f"q_{nonce}")

cat_col  = st.sidebar.selectbox("Category column", [""] + txt_cols, key=f"cat_col_{nonce}")
cat_vals = None
if cat_col:
    uni = sorted([str(x) for x in df[cat_col].dropna().unique()])
    cat_vals = st.sidebar.multiselect("Category values", uni, key=f"cat_vals_{nonce}")

date_col = st.sidebar.selectbox("Date column", [""] + date_cols, key=f"date_col_{nonce}")
date_rng = None
if date_col:
    dmin = pd.to_datetime(df[date_col]).min()
    dmax = pd.to_datetime(df[date_col]).max()
    date_rng = st.sidebar.date_input(
        "Date range",
        value=(dmin.date(), dmax.date()),
        min_value=dmin.date(), max_value=dmax.date(),
        key=f"date_rng_{nonce}",
    )

metric_col   = st.sidebar.selectbox("Metric (for KPIs/Charts)", [""] + num_cols, key=f"metric_col_{nonce}")
metric_range = None
if metric_col:
    lo = float(pd.to_numeric(df[metric_col], errors="coerce").min() or 0.0)
    hi = float(pd.to_numeric(df[metric_col], errors="coerce").max() or 0.0)
    metric_range = st.sidebar.slider(
        "Metric range", min_value=0.0, max_value=max(1.0, hi),
        value=(lo, hi), key=f"metric_range_{nonce}"
    )

# Apply filters
dff = df.copy()
if text_col and q:
    dff = dff[dff[text_col].astype(str).str.contains(q, case=False, na=False)]
if cat_col and cat_vals and len(cat_vals) < len(dff[cat_col].dropna().unique()):
    dff = dff[dff[cat_col].astype(str).isin(cat_vals)]
if date_col and date_rng:
    start = pd.to_datetime(date_rng[0])
    end   = pd.to_datetime(date_rng[1])
    dff = dff[(pd.to_datetime(dff[date_col]) >= start) & (pd.to_datetime(dff[date_col]) <= end)]
if metric_col and metric_range:
    lo, hi = metric_range
    vals = pd.to_numeric(dff[metric_col], errors="coerce")
    dff = dff[(vals >= lo) & (vals <= hi)]

st.sidebar.divider()
q_all = st.sidebar.text_input("Global search (any column)", key=f"q_all_{nonce}")
if q_all:
    mask = dff.astype(str).apply(lambda s: s.str.contains(q_all, case=False, na=False))
    dff = dff[mask.any(axis=1)]

# ============================== 3) KPIs ==============================
c1, c2, c3 = st.columns(3)
c1.metric("Rows", f"{len(dff):,}")
if metric_col:
    total = pd.to_numeric(dff[metric_col], errors="coerce").sum()
    median = pd.to_numeric(dff[metric_col], errors="coerce").median()
    c2.metric(f"Sum {metric_col}", f"{total:,.2f}")
    c3.metric(f"Median {metric_col}", f"{median:,.2f}")

# ============================== Columns / Sort & Paginate ==============================
st.sidebar.markdown("### Columns")
needle = st.sidebar.text_input("Find column name", key=f"needle_{nonce}")
available_cols = [c for c in dff.columns if needle.lower() in c.lower()]
default_cols   = available_cols[:30]
visible_cols   = st.sidebar.multiselect(
    "Columns to display/export", available_cols, default=default_cols, key=f"visible_cols_{nonce}"
)

st.sidebar.markdown("### Sort & paginate")
sort_choices = [""] + list((visible_cols or list(dff.columns)))
sort_col  = st.sidebar.selectbox("Sort by", sort_choices, key=f"sort_col_{nonce}")
sort_dir  = st.sidebar.radio("Order", ["Ascending","Descending"], horizontal=True, key=f"sort_dir_{nonce}")
page_size = st.sidebar.selectbox("Rows per page", [25,50,100,250,1000], key=f"page_size_{nonce}")
st.session_state.page_size = int(page_size)

extra = ["_id"] if "_id" in dff.columns and "_id" not in (visible_cols or []) else []
view = dff[(visible_cols or dff.columns.tolist()) + extra] if visible_cols else dff

if sort_col:
    view_sorted = view.sort_values(by=sort_col, ascending=(sort_dir=="Ascending"), kind="mergesort")
else:
    view_sorted = view

total_rows = len(view_sorted)
total_pages = max(1, math.ceil(total_rows / st.session_state.page_size))
st.session_state.page = min(max(1, st.session_state.page), total_pages)
page = st.session_state.page

start_i = (page - 1) * st.session_state.page_size
end_i   = min(start_i + st.session_state.page_size, total_rows)
page_view = view_sorted.iloc[start_i:end_i]

# ============================== 4) Charts ==============================
tabs = st.tabs(["Time series", "By category", "Table / Edit"])
con = register_duck(dff)

with tabs[0]:
    if metric_col and date_col:
        ts = con.execute(
            f'SELECT "{date_col}" AS date_, SUM("{metric_col}") AS metric_ '
            "FROM data GROUP BY 1 ORDER BY 1"
        ).df()
        if ts.empty:
            st.info("No data for current filters.")
        else:
            fig = px.line(ts, x="date_", y="metric_", title=f"{metric_col} over time",
                          labels={"date_": date_col, "metric_": metric_col}, height=320)
            fig.update_layout(yaxis_tickformat="~s")
            st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Pick a Date + Metric to see a time series.")

with tabs[1]:
    if metric_col and cat_col:
        byc = con.execute(
            f'SELECT "{cat_col}" AS category_, SUM("{metric_col}") AS metric_ '
            "FROM data GROUP BY 1 ORDER BY 2 DESC"
        ).df()
        if byc.empty:
            st.info("No data for current filters.")
        else:
            fig = px.bar(byc, x="category_", y="metric_", title=f"{metric_col} by {cat_col}",
                         labels={"category_": cat_col, "metric_": metric_col}, height=320)
            fig.update_layout(yaxis_tickformat="~s")
            st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Pick a Category + Metric to see a bar chart.")

# ============================== 5) CRUD ==============================
with tabs[2]:
    st.subheader("Edit data (CRUD)")
    st.caption(
        f"Showing rows {start_i + 1}–{end_i} of {total_rows} "
        f"(page {page}/{total_pages}). Use sorting & pagination in the sidebar."
    )

    # Pager
    p1, p2, p3 = st.columns([1,2,1])
    if p1.button("◀ Prev", disabled=(page <= 1)):
        st.session_state.page = max(1, page - 1); st.rerun()
    p2.markdown(f"<div style='text-align:center'>Page <b>{page}</b> / <b>{total_pages}</b></div>", unsafe_allow_html=True)
    if p3.button("Next ▶", disabled=(page >= total_pages)):
        st.session_state.page = min(total_pages, page + 1); st.rerun()

    # ----- Stable Add Row (form)
    with st.expander("➕ Add a row", expanded=False):
        st.caption("Safer than typing into the editor’s blank row on long/sorted tables.")
        form_cols = [c for c in (visible_cols or list(df.columns)) if c not in ["_id", "Delete"]]

        with st.form("add_row_form", clear_on_submit=True):
            inputs: Dict[str, Any] = {}
            left, right = st.columns(2)
            for i, c in enumerate(form_cols):
                host = left if i % 2 == 0 else right
                if pd.api.types.is_datetime64_any_dtype(df[c]):
                    val = host.date_input(c, value=None)
                elif pd.api.types.is_numeric_dtype(df[c]):
                    val = host.text_input(c, value="", placeholder="e.g. 123.45")
                else:
                    val = host.text_input(c, value="")
                inputs[c] = val
            submitted = st.form_submit_button("Add")

        if submitted:
            new = {}
            for c, v in inputs.items():
                if v == "" or v is None:
                    new[c] = pd.NA
                else:
                    # try numeric first
                    if pd.api.types.is_numeric_dtype(df[c]):
                        new[c] = pd.to_numeric(v, errors="coerce")
                    elif "date" in str(type(v)).lower():
                        new[c] = pd.Timestamp(v)
                    else:
                        new[c] = v
            new["_id"] = st.session_state.next_id
            st.session_state.next_id += 1

            st.session_state.last_df = st.session_state.df.copy()
            st.session_state.df = pd.concat(
                [st.session_state.df, pd.DataFrame([new])],
                ignore_index=True
            )
            st.toast("Row added.", icon="➕")
            st.session_state.page = math.ceil(len(st.session_state.df) / st.session_state.page_size)
            st.rerun()

    # Editor for update/delete only
    edit_view = page_view.copy()
    if "Delete" not in edit_view.columns:
        edit_view["Delete"] = False

    edited = st.data_editor(
        edit_view,
        num_rows="fixed",           # no dynamic blank row
        use_container_width=True,
        hide_index=True,
        key="editor_table",
        column_config={"_id": st.column_config.NumberColumn("_id", help="internal id", disabled=True)},
        height=420,
    )

    cA, cB, cC, cD = st.columns([1,1,1,2])
    apply_btn = cA.button("Apply changes", type="primary")
    undo_btn  = cB.button("Undo last change")
    cC.download_button(
        "Download page CSV",
        edited.drop(columns=["Delete"], errors="ignore").to_csv(index=False).encode(),
        "page.csv","text/csv",
    )
    cD.download_button(
        "Download filtered CSV",
        view_sorted.to_csv(index=False).encode(),
        "filtered.csv","text/csv",
    )

    if undo_btn and st.session_state.last_df is not None:
        st.session_state.df = st.session_state.last_df.copy()
        st.toast("Reverted to last saved dataset.", icon="↩️"); st.rerun()

    if apply_btn:
        base = st.session_state.df.set_index("_id").copy()
        st.session_state.last_df = base.reset_index().copy()

        # Deletes
        to_delete = edited.loc[edited["Delete"] == True, "_id"].dropna().astype(int).tolist()  # noqa: E712
        if to_delete:
            base = base.drop(index=[rid for rid in to_delete if rid in base.index], errors="ignore")

        # Updates (no inserts here—use the form above)
        updates = edited[edited["_id"].notna()].drop(columns=["Delete"]).set_index("_id")
        if not updates.empty:
            cols_to_update = [c for c in updates.columns if c in base.columns]
            base.update(updates[cols_to_update])

        st.session_state.df = base.reset_index()
        st.toast("Changes applied.", icon="✅"); st.rerun()

# ============================== 6) Export (full) ==============================
st.download_button(
    "Download current dataset (all rows)",
    st.session_state.df.to_csv(index=False).encode(),
    "dataset.csv","text/csv",
)
