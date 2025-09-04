# CSV Studio — DuckDB Edition

[![Streamlit App](https://img.shields.io/badge/Live%20App-Streamlit-black?logo=streamlit)](https://csv-studio-duckdb.streamlit.app/)
![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?logo=python&logoColor=white)
![DuckDB](https://img.shields.io/badge/Powered_by-DuckDB-fff?logo=duckdb&logoColor=000)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

A compact, browser-based **CSV workbench**. Upload a file or paste a CSV URL, filter and chart it, edit rows, and export—fast.

---

## Capabilities

- **Load** CSV/TSV/XLSX or **fetch** a public CSV by URL
- **Filter & search:** text, category, date range, global search, metric range
- **KPIs & charts:** sum/median, **time series** (date+metric), **by category** (metric by dimension)
- **CRUD table editor:** add/edit/delete rows with safe internal `_id`
- **Sort & paginate** large tables (25/50/100/250/1000 rows)
- **Export** filtered view or full dataset
- **One-click resets:** clear filters, clear data, reset app

---

## Features (at a glance)

- Minimal **black & gold** UI with clean, glossy touches
- In-memory **pandas** DataFrame with **DuckDB** for fast aggregations
- Works great for ad-hoc cleaning, quick analytics, and sharing filtered CSVs

---

## Credits

- **[Streamlit](https://streamlit.io/):** UI framework
- **[DuckDB](https://duckdb.org/):** analytics engine
- **[Plotly](https://plotly.com/python/):** interactive charts

---

## License

Released under the **MIT License**. See [`LICENSE`](LICENSE).
