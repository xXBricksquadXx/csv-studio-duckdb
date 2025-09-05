# CSV Studio — DuckDB Edition

[![Live App](https://img.shields.io/badge/Live_App-Streamlit-ff4b4b?logo=streamlit&logoColor=white)](https://csv-studio-duckdb.streamlit.app/)
![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?logo=python&logoColor=white)
![DuckDB](https://img.shields.io/badge/Powered_by-DuckDB-fff?logo=duckdb&logoColor=000)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

A compact, browser-based **CSV workbench** powered by **DuckDB + Streamlit**.  
Upload a file or paste a CSV URL, filter + chart it, **edit rows**, and export—fast.

<p align="center">
  <a href="https://csv-studio-duckdb.streamlit.app/">Live demo</a> ·
  <a href="https://github.com/xXBricksquadXx/csv-studio-duckdb">Source</a>
</p>

<p align="center">
  <img src="docs/screenshot-main.png" alt="CSV Studio — minimal CSV workbench with filters, CRUD editor, pagination and charts" width="100%">
</p>

---

[![Use this template](https://img.shields.io/badge/Use_this_template-2ea44f?logo=github)](https://github.com/xXBricksquadXx/csv-studio-duckdb/generate)

## Features

- **Load data**
  - CSV / TSV / XLSX upload or **fetch** a public CSV by URL
  - “**Load sample**” button for instant play
- **Filter & search**
  - Global text search across all string columns
  - Category pickers + multi-select values
  - Metric range slider (int/float/**boolean-safe**, constant/NaN-safe)
  - Optional date column for time series
- **Charts & KPIs**
  - Sum & median KPIs
  - **Time series** (date + metric) and **By category** (metric grouped by dimension)
- **Table / Edit (CRUD)**
  - Fast, paginated editor with **sticky headers**
  - Add / update / delete rows (stable internal `_id`)
  - **Quick filter** input above the table
- **Export**
  - Page CSV, filtered CSV, or **full dataset (Parquet)**
- **Ergonomics**
  - **Noir+Gold** premium dark theme (w/ Light toggle)
  - **Share this view**: saves filters to URL params for easy linking
  - **Clear data** & **Clear filters** one-click actions

---

## Quickstart

```bash
# 1) Create & activate a venv (Windows PowerShell shown)
python -m venv .venv
.\.venv\Scripts\Activate.ps1

# 2) Install dependencies
pip install -r requirements.txt  # streamlit, duckdb, pandas, plotly, openpyxl

# 3) Run the app
python -m streamlit run app.py
```

---

## How to use

Load data from the sidebar: upload CSV/TSV/XLSX, paste a CSV URL, or press Load sample.

Use Global search and Category / Metric filters to shape your view.

Switch between Charts and Table / Edit tabs.

Table tab has a Quick filter box (any column).

Use Prev / Next to paginate large data.

Click Apply changes to persist edits back to the in-memory dataset.

Export page CSV, filtered CSV, or the full dataset as Parquet.

Press 🔗 Share this view to push filters into the URL and share the exact state.

---

## Why DuckDB?

DuckDB is an in-process analytical database—perfect for ad-hoc querying without maintaining a server. Query performance stays snappy in the browser app via Python bindings and in-memory tables.

---

## Good test CSV URLs

```pwsh
Chicago community areas (small):
https://raw.githubusercontent.com/datadesk/census-data-1/master/data/education.csv

California cities (medium):
https://raw.githubusercontent.com/plotly/datasets/master/2014_usa_states.csv

NYC 311 sample (larger):
https://raw.githubusercontent.com/vaibhavk97/NYC-311-Data-Analysis/master/311_Service_Requests_from_2010_to_Present.csv
```

Any public raw CSV link works (GitHub “Raw” URLs are great). Very large files depend on your machine/host memory.

---

## License

MIT © 2025 — Feel free to fork, adapt, and ship.
