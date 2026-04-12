 
from flask import Flask, jsonify, request, render_template, send_from_directory
from flask_cors import CORS
import pandas as pd
import numpy as np
from scipy.stats import linregress
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import warnings, os, json
warnings.filterwarnings('ignore')

app = Flask(__name__, template_folder='templates', static_folder='static')
CORS(app)

 

 
MS = {}   

def train_model():
    print("[STARTUP] Loading data and training model...")

    file_path = "data/Mumbai_UHI_7day_2015_2025.csv"
    df = pd.read_csv(file_path)
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values('date').drop_duplicates(subset='date')
    df.set_index('date', inplace=True)

    cols = ['NDVI', 'Urban_LST', 'Rural_LST', 'UHI']
    for col in cols:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    df[cols] = df[cols].interpolate(method='time', limit_direction='both')

    df['month']        = df.index.month
    df['year']         = df.index.year
    df['day_of_year']  = df.index.dayofyear
    df['LST_diff']     = df['Urban_LST'] - df['Rural_LST']
    df['UHI_lag1']     = df['UHI'].shift(1)
    df['UHI_lag4']     = df['UHI'].shift(4)
    df['UHI_rolling4'] = df['UHI'].rolling(4, min_periods=1).mean()
    df['month_sin']    = np.sin(2 * np.pi * df['month'] / 12)
    df['month_cos']    = np.cos(2 * np.pi * df['month'] / 12)
    df['doy_sin']      = np.sin(2 * np.pi * df['day_of_year'] / 365)
    df['doy_cos']      = np.cos(2 * np.pi * df['day_of_year'] / 365)
    df.dropna(inplace=True)

    yearly = df.groupby('year')[['NDVI', 'Urban_LST', 'Rural_LST', 'UHI']].mean()
    def fit_trend(s):
        x = np.array(s.index, dtype=float)
        sl, ic, r, _, _ = linregress(x, s.values)
        return sl, ic, r**2

    ndvi_s,  ndvi_b,  _  = fit_trend(yearly['NDVI'])
    ulst_s,  ulst_b,  _  = fit_trend(yearly['Urban_LST'])
    rlst_s,  rlst_b,  _  = fit_trend(yearly['Rural_LST'])
    uhi_s,   uhi_b,   uhi_r2 = fit_trend(yearly['UHI'])

    df['time_idx']     = np.arange(len(df), dtype=float)
    trend_model        = LinearRegression()
    trend_model.fit(df[['time_idx']], df['UHI'])
    df['UHI_trend']    = trend_model.predict(df[['time_idx']])
    df['UHI_detrended']= df['UHI'] - df['UHI_trend']

    features = [
        'NDVI', 'LST_diff', 'Urban_LST', 'Rural_LST',
        'month_sin', 'month_cos', 'doy_sin', 'doy_cos',
        'year', 'UHI_lag1', 'UHI_lag4', 'UHI_rolling4'
    ]

    X = df[features]
    y = df['UHI_detrended']
    split = int(len(df) * 0.8)
    X_train, X_test = X.iloc[:split], X.iloc[split:]
    y_train, y_test = y.iloc[:split], y.iloc[split:]

    rf = RandomForestRegressor(n_estimators=300, max_depth=12,
                               min_samples_leaf=3, max_features=0.7,
                               random_state=42, n_jobs=-1)
    rf.fit(X_train, y_train)

    test_tidx = df['time_idx'].iloc[split:].values.reshape(-1, 1)
    rf_res    = rf.predict(X_test)
    trend_t   = trend_model.predict(test_tidx).ravel()
    rf_preds  = rf_res + trend_t
    y_actual  = df['UHI'].iloc[split:].values

    rmse = float(np.sqrt(mean_squared_error(y_actual, rf_preds)))
    mae  = float(mean_absolute_error(y_actual, rf_preds))
    r2   = float(r2_score(y_actual, rf_preds))

    feat_imp = dict(zip(features, rf.feature_importances_.tolist()))
    monthly_stats = df.groupby('month')[['NDVI', 'Urban_LST', 'Rural_LST']].mean()

    MS['df']           = df
    MS['rf']           = rf
    MS['trend_model']  = trend_model
    MS['features']     = features
    MS['split']        = split
    MS['metrics']      = {'rmse': rmse, 'mae': mae, 'r2': r2}
    MS['feat_imp']     = feat_imp
    MS['monthly_stats']= monthly_stats
    MS['slopes']       = {'ndvi': ndvi_s, 'urban_lst': ulst_s,
                          'rural_lst': rlst_s, 'uhi': uhi_s,
                          'uhi_b': uhi_b, 'uhi_r2': uhi_r2}
    MS['test_preds']   = rf_preds.tolist()
    MS['test_actual']  = y_actual.tolist()
    MS['test_dates']   = [d.strftime('%Y-%m-%d') for d in df.index[split:]]
    MS['yearly']       = yearly.reset_index().to_dict(orient='records')
    MS['last_time_idx']= float(df['time_idx'].iloc[-1])
    MS['last_date']    = df.index[-1]
    MS['last_year']    = int(df.index[-1].year)

    print(f"[STARTUP] Done. RMSE={rmse:.4f}  R²={r2:.4f}")


def run_forecast(end_year):
    df           = MS['df']
    rf           = MS['rf']
    trend_model  = MS['trend_model']
    monthly_stats= MS['monthly_stats']
    slopes       = MS['slopes']
    last_year    = MS['last_year']
    last_date    = MS['last_date']
    last_tidx    = MS['last_time_idx']

    future_steps = (end_year - last_year) * 52
    future_preds, ci_low, ci_high, dates = [], [], [], []
    recent_uhi = list(df['UHI'].values[-8:])

    for i in range(1, future_steps + 1):
        nd         = last_date + pd.Timedelta(weeks=i)
        month      = nd.month
        year       = nd.year
        doy        = nd.dayofyear
        yr_delta   = year - last_year

        NDVI      = monthly_stats.loc[month, 'NDVI']      + slopes['ndvi']      * yr_delta
        Urban_LST = monthly_stats.loc[month, 'Urban_LST'] + slopes['urban_lst'] * yr_delta
        Rural_LST = monthly_stats.loc[month, 'Rural_LST'] + slopes['rural_lst'] * yr_delta
        LST_diff  = Urban_LST - Rural_LST

        row = np.array([[
            NDVI, LST_diff, Urban_LST, Rural_LST,
            np.sin(2*np.pi*month/12), np.cos(2*np.pi*month/12),
            np.sin(2*np.pi*doy/365),  np.cos(2*np.pi*doy/365),
            year,
            recent_uhi[-1],
            recent_uhi[-4] if len(recent_uhi) >= 4 else recent_uhi[0],
            np.mean(recent_uhi[-4:])
        ]])

        tree_preds = np.array([t.predict(row)[0] for t in rf.estimators_])
        rf_mean    = tree_preds.mean()
        rf_std     = tree_preds.std()
        trend_val  = trend_model.predict([[last_tidx + i]])[0]
        pred       = rf_mean + trend_val

        future_preds.append(round(float(pred), 4))
        ci_low.append(round(float(pred - 1.96 * rf_std), 4))
        ci_high.append(round(float(pred + 1.96 * rf_std), 4))
        dates.append(nd.strftime('%Y-%m-%d'))
        recent_uhi.append(pred)
        recent_uhi = recent_uhi[-8:]

    return {'dates': dates, 'predicted': future_preds,
            'ci_low': ci_low, 'ci_high': ci_high}


 

 

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/historical')
def historical():
    df = MS['df']
    start = request.args.get('start', str(df.index.min().date()))
    end   = request.args.get('end',   str(df.index.max().date()))
    mask  = (df.index >= start) & (df.index <= end)
    sub   = df.loc[mask]

    return jsonify({
        'dates':      [d.strftime('%Y-%m-%d') for d in sub.index],
        'uhi':        sub['UHI'].round(4).tolist(),
        'urban_lst':  sub['Urban_LST'].round(4).tolist(),
        'rural_lst':  sub['Rural_LST'].round(4).tolist(),
        'ndvi':       sub['NDVI'].round(4).tolist(),
    })

@app.route('/api/forecast')
def forecast():
    end_year = int(request.args.get('end_year', 2040))
    end_year = max(MS['last_year'] + 1, min(end_year, 2060))
    result = run_forecast(end_year)
    return jsonify(result)

@app.route('/api/metrics')
def metrics():
    return jsonify({
        **MS['metrics'],
        'n_train': MS['split'],
        'n_test':  len(MS['df']) - MS['split'],
        'n_total': len(MS['df']),
        'date_range': {
            'start': MS['df'].index.min().strftime('%Y-%m-%d'),
            'end':   MS['df'].index.max().strftime('%Y-%m-%d'),
        },
        'uhi_trend_slope': round(MS['slopes']['uhi'], 4),
        'uhi_trend_r2':    round(MS['slopes']['uhi_r2'], 4),
    })

@app.route('/api/feature_importance')
def feature_importance():
    fi = sorted(MS['feat_imp'].items(), key=lambda x: x[1], reverse=True)
    return jsonify({'features': [f for f,_ in fi],
                    'importance': [round(v,4) for _,v in fi]})

@app.route('/api/test_predictions')
def test_predictions():
    return jsonify({
        'dates':     MS['test_dates'],
        'actual':    [round(v,4) for v in MS['test_actual']],
        'predicted': [round(v,4) for v in MS['test_preds']],
    })

@app.route('/api/seasonal')
def seasonal():
    df = MS['df']
    monthly = df.groupby('month')['UHI'].agg(['mean','std','min','max']).reset_index()
    months  = ['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec']
    return jsonify({
        'months': months,
        'mean':   monthly['mean'].round(4).tolist(),
        'std':    monthly['std'].round(4).tolist(),
        'min':    monthly['min'].round(4).tolist(),
        'max':    monthly['max'].round(4).tolist(),
    })

@app.route('/api/yearly_trend')
def yearly_trend():
    yearly = MS['yearly']
    slopes = MS['slopes']
    return jsonify({'yearly': yearly, 'slopes': slopes})

@app.route('/api/heatmap_data')
def heatmap_data():
    df = MS['df']
    pivot = df.pivot_table(values='UHI', index='year', columns='month', aggfunc='mean')
    months = ['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec']
    return jsonify({
        'years':  pivot.index.tolist(),
        'months': months,
        'values': [[round(v,2) if not np.isnan(v) else None
                    for v in row] for row in pivot.values.tolist()]
    })

 
if __name__ == '__main__':
    train_model()
    app.run(debug=True, port=5000)