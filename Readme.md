# Urban Heat Island Analysis — Mumbai

A full-stack data science project studying how Mumbai's urban landscape raises temperatures compared to surrounding rural areas, using satellite imagery from 2015 to 2025 and a machine learning model to forecast what the next 15 years might look like.

---

## Why this project

Mumbai is one of the fastest-urbanising cities in Asia. Between construction, population density, reduced tree cover, and the heat-retaining properties of concrete and asphalt, the city's core runs significantly hotter than its outskirts — a phenomenon called the **Urban Heat Island (UHI) effect**.

This isn't just a number on a graph. A 5–8°C difference between Dharavi and the Thane periphery has real consequences: higher energy demand, heat stress on vulnerable populations, and compounding effects as climate change raises baselines across the board.

This project started as an attempt to quantify that difference systematically, using freely available Landsat satellite data, and then ask: *if current trends continue, how bad does it get by 2040?*

---

## What's inside

```
UHI-Analysis-Mumbai/
│
├── data
    ├──Mumbai_UHI_7day_2015_2025.csv   # 10 years of weekly satellite data
├── outputs
    ├── UHI_Full_Featured_Dataset.csv         # All satellite data at one place 
    ├── UHI_Future_Predictions_2026_2030.csv  # Predicted values from 2026-2030
    ├── UHI_Future_Predictions_Improved.csv   # Future Prediction Values
├── app.py                        
├── requirements.txt                
│
└── templates/
    └── index.html                  #
```

---

## The data

Everything comes from **Landsat 8 and Landsat 9** imagery, processed through Google Earth Engine and exported as weekly composites. Three core measurements per week:

| Column | What it is |
|--------|-----------|
| `Urban_LST` | Land Surface Temperature (°C) averaged over Mumbai's urban core |
| `Rural_LST` | LST averaged over surrounding rural/semi-rural zones |
| `NDVI` | Normalized Difference Vegetation Index — a measure of green cover |
| `UHI` | Simply `Urban_LST − Rural_LST`. The headline number. |

169 of the ~497 weekly records had missing values — mostly cloud-contaminated imagery during monsoon season. These were filled using time-based linear interpolation rather than dropping rows, to preserve the seasonal structure.

---

## The model

### Why not just Random Forest?

The obvious choice for a regression problem like this is Random Forest, and it works well in hindsight (predicting test data from the same time period). The problem is **extrapolation**: Random Forest can only return values within the range it was trained on. Ask it to predict 2035 and it will give you something that looks like 2021, just repeated.

To fix this, the model was split into two parts:

**Step 1 — Linear Regression on the long-term trend**

A simple linear model is fit on yearly UHI averages to capture the slow upward drift (+0.099°C per year). This component is responsible for the rising baseline over the forecast window.

**Step 2 — Random Forest on the residual**

Once the trend is removed, what's left is the seasonal variation and noise. Random Forest handles this part well because it's pattern-matching within a bounded range. At forecast time, the two components are added back together.

```
Final prediction = RF(detrended residual) + Linear trend component + Seasonal mean
```

### Feature engineering

Beyond the raw columns, the model uses:

- **Cyclical time encodings** — month and day-of-year expressed as sine/cosine pairs, so December and January are treated as adjacent rather than opposite ends of a number line
- **Lag features** — UHI values from 1, 4, 8, and 13 weeks prior, giving the model a memory of recent conditions
- **Rolling statistics** — 4-week and 8-week rolling mean and standard deviation
- **LST_diff** — explicitly including `Urban_LST − Rural_LST` as a feature alongside the raw values reduces the model's tendency to just re-derive UHI from the two temperature columns directly (data leakage)
- **Year** — added as a feature so the RF can learn that 2022 patterns differ from 2016 within the training window

### Future feature projection

For forecasting, NDVI and LST values needed to be projected forward too — you can't just feed 2019 seasonal averages when predicting 2036. Each variable's historical yearly slope (calculated via `scipy.stats.linregress`) is used to nudge the monthly baseline forward year by year. So January 2036's Urban_LST input is January's historical mean plus (years elapsed × rate of change).

### Confidence intervals

Rather than a fixed ±1.96σ, each of the 500 trees in the Random Forest makes its own prediction. The spread across those 500 predictions gives a data-driven uncertainty band — wider in future years where the model is less certain, narrower for near-term forecasts.

### Results

| Metric | Value |
|--------|-------|
| RMSE | 1.07°C |
| MAE | 0.75°C |
| R² | 0.548 |
| UHI trend | +0.099°C per year |

R² of 0.548 means the model explains about 55% of the week-to-week variance. The remainder is driven by factors not in the dataset — specific weather events, construction activity, rainfall anomalies. For a 10-year weekly time series with 34% imputed values, this is reasonable.

---

## Forecast findings

| Period | Projected mean UHI | 95% CI |
|--------|-------------------|--------|
| 2026–2030 | 6.57°C | [2.4, 10.8] |
| 2031–2035 | 8.05°C | [3.2, 13.0] |
| 2036–2040 | 9.28°C | [3.8, 14.7] |

The wide confidence intervals are honest — 15-year forecasts from a weekly dataset carry real uncertainty. The trend direction is consistent with Mumbai's known urbanisation trajectory, but the exact values depend heavily on future development patterns, green cover policy, and monsoon variability.

---

## Seasonal patterns

A few things stood out when looking at the Year × Month heatmap:

- **March to May** consistently shows the highest UHI intensity. Pre-monsoon heat, low humidity, and no cloud cover combine to maximise the urban-rural temperature gap.
- **June to September** the UHI drops sharply. Monsoon cloud cover reduces incoming solar radiation, soil moisture rises in rural areas, and NDVI increases from vegetation green-up.
- **2020** shows anomalous behaviour — likely a COVID lockdown signal, with reduced traffic and industrial activity compressing the urban heat signature temporarily.

---

## The web dashboard

The frontend is a single HTML file that calls a FastAPI backend. No React, no build pipeline — just vanilla JavaScript with Chart.js and Leaflet.

**Four pages:**

- **Dashboard** — the main forecast chart, test set accuracy, and feature importance
- **Forecast** — long-range prediction with annual/weekly toggle and 5-year summary table
- **Analysis** — monthly seasonality, Urban vs Rural LST trends, and the Year × Month heatmap
- **Map** — 12 Mumbai neighbourhoods on a Leaflet map with circle markers sized by UHI intensity, plus a heatmap layer

The date range filter and forecast horizon selector in the sidebar all hit the FastAPI backend in real time — changing the historical window or switching from 2035 to 2050 fires a new API call and redraws the charts without a page reload.

---

## Running it locally

**1. Clone the repo**
```bash
git clone https://github.com/rajatsurana19/UHI-Analysis-Mumbai.git
cd UHI-Analysis-Mumbai
```

**2. Install dependencies**
```bash
pip install -r requirements.txt
```

**3. Start the backend**
```bash
uvicorn main:app --reload --port 8000
```

The first startup takes 2–3 minutes — the model trains on launch. You'll see the RMSE and R² printed once it's ready.

**4. Open the dashboard**

Go to `http://localhost:8000` in your browser. Everything else is handled by the running server.

---

## API endpoints

If you want to hit the backend directly without the frontend:

| Endpoint | What it returns |
|----------|----------------|
| `GET /api/historical?start=2018-01-01&end=2022-12-31` | Historical UHI, LST, NDVI for a date range |
| `GET /api/forecast?end_year=2040` | Weekly predictions with confidence intervals |
| `GET /api/metrics` | RMSE, MAE, R², trend slope, CV scores |
| `GET /api/seasonal` | Monthly mean, std, min, max UHI |
| `GET /api/heatmap_data` | Year × Month pivot table |
| `GET /api/feature_importance` | Top 20 feature importances from RF |
| `GET /api/yearly_trend` | Annual averages and regression slopes |
| `GET /api/model_comparison` | Individual base learner R² vs ensemble |
| `GET /api/health` | Quick check that the model is loaded |

Interactive docs at `http://localhost:8000/docs` once the server is running.

---

## Dependencies

| Package | Purpose |
|---------|---------|
| `fastapi` + `uvicorn` | Backend web server |
| `pandas` + `numpy` | Data wrangling |
| `scikit-learn` | Linear Regression, Random Forest, metrics |
| `scipy` | Trend line fitting (`linregress`) |
| Chart.js | Frontend charts |
| Leaflet.js | Interactive Mumbai map |

---

## Known limitations

- The 169 missing rows (34% of NDVI and UHI records) were interpolated. For a cleaner dataset, these should be manually verified against raw Landsat imagery.
- The spatial component is simulated — neighbourhood-level UHI values on the map are derived from zonal estimates, not pixel-level ground truth for each locality.
- The model has no knowledge of future policy decisions, infrastructure changes, or rainfall anomalies. The forecast assumes recent trends continue linearly.
- R² of 0.548 leaves room for improvement. Adding atmospheric variables (humidity, wind speed) and finer spatial resolution would likely help.

---

## Possible extensions

- Pull live Landsat data via Google Earth Engine Python API and update the CSV automatically
- Add a pixel-level spatial layer using actual raster data instead of zone estimates
- Incorporate IMD rainfall data as an additional predictor
- Build a comparison between Mumbai and another Indian metro (Delhi, Pune, Bengaluru)

---

## Author

Made by **Rajat Surana**

[![LinkedIn](https://img.shields.io/badge/LinkedIn-rajat--surana-0A66C2?style=flat-square&logo=linkedin&logoColor=white)](https://linkedin.com/in/rajat-surana)
[![GitHub](https://img.shields.io/badge/GitHub-rajatsurana19-181717?style=flat-square&logo=github&logoColor=white)](https://github.com/rajatsurana19)
