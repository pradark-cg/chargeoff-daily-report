"""
Chargeoff Daily Forecast
Runs for today's date, queries Snowflake, computes empirical flow rates,
and writes an HTML report to docs/index.html (served by GitHub Pages).
"""
import os, sys, json
import pandas as pd
import numpy as np
from datetime import date
from pathlib import Path
from snowflake.snowpark import Session

try:
    # ── 1. Connect ────────────────────────────────────────────────────────────────
    session = Session.builder.configs({
        'account':   os.environ.get('SF_ACCOUNT'),
        'user':      os.environ.get('SF_USER'),
        'password':  os.environ.get('SF_PASSWORD'),
        'role':      os.environ.get('SF_ROLE'),
        'warehouse': os.environ.get('SF_WAREHOUSE'),
        'database':  os.environ.get('SF_DATABASE'),
        'schema':    os.environ.get('SF_SCHEMA'),
    }).create()
    print('✓ Connected to Snowflake.')
except Exception as e:
    print(f'✗ Connection failed: {e}', file=sys.stderr)
    sys.exit(1)

# ── 2. Run date — use latest available date in dailybalance ───────────────────
try:
    max_date_row = session.sql(
        "select max(cast(date as date)) as max_date from creditgenie.public.dailybalance"
    ).to_pandas()
    if max_date_row.empty or max_date_row['MAX_DATE'].iloc[0] is None:
        raise ValueError("No data in dailybalance")
    RUN_DATE = pd.Timestamp(max_date_row['MAX_DATE'].iloc[0])
    print(f'✓ Run Date: {RUN_DATE.date()}')
except Exception as e:
    print(f'✗ Failed to get run date: {e}', file=sys.stderr)
    sys.exit(1)

# ── 3. Calibration window (last 6 fully mature CO months) ─────────────────────
run_day         = RUN_DATE.day
CALIB_START     = '2025-10-01'
CALIB_END       = '2026-03-01'
calib_co_months = pd.date_range(CALIB_START, CALIB_END, freq='MS')

calib_configs = []
all_calib_dates = set()
for co_start in calib_co_months:
    co_end = co_start + pd.offsets.MonthEnd(0)
    r3 = (co_start - pd.DateOffset(months=3)).replace(day=run_day)
    r2 = (co_start - pd.DateOffset(months=2)).replace(day=run_day)
    r1 = (co_start - pd.DateOffset(months=1)).replace(day=run_day)
    all_calib_dates.update([r3, r2, r1])
    calib_configs.append({
        'co_month': co_start.strftime('%Y-%m'),
        'co_start': co_start, 'co_end': co_end,
        'run_3mo': pd.Timestamp(r3),
        'run_2mo': pd.Timestamp(r2),
        'run_1mo': pd.Timestamp(r1),
    })

# ── 4. Query calibration snapshots ────────────────────────────────────────────
calib_date_list = ', '.join(d.strftime("'%Y-%m-%d'") for d in sorted(all_calib_dates))
df_calib = session.sql(f"""
    select d.advanceid, d.origday,
        cast(d.date as date) as snapshot_date,
        dateadd(day, 120 - d.origday, cast(d.date as date)) as chargeoff_date,
        d.principalbalance
    from creditgenie.public.dailybalance d
    left join creditgenie.ddb_prod.advance a on d.advanceid = a.id
    where d.date in ({calib_date_list})
      and d.origday < 120 and d.principalbalance > 0
      and a.status not in ('Failed','Cancelled','Canceled')
""").to_pandas()
df_calib.columns = df_calib.columns.str.lower()
df_calib['snapshot_date']  = pd.to_datetime(df_calib['snapshot_date'])
df_calib['chargeoff_date'] = pd.to_datetime(df_calib['chargeoff_date'])
df_calib['principalbalance'] = df_calib['principalbalance'].astype(float)
print(f'Calibration rows: {len(df_calib):,}')

# ── 5. Query actuals ──────────────────────────────────────────────────────────
calib_end_str = (pd.Timestamp(CALIB_END) + pd.offsets.MonthEnd(0)).strftime('%Y-%m-%d')
df_actuals = session.sql(f"""
    select d.advanceid, cast(d.date as date) as snapshot_date, d.principalbalance
    from creditgenie.public.dailybalance d
    left join creditgenie.ddb_prod.advance a on d.advanceid = a.id
    where d.origday = 120 and d.principalbalance > 0
      and a.status not in ('Failed','Cancelled','Canceled')
      and cast(d.date as date) >= '{CALIB_START}'
      and cast(d.date as date) <= '{calib_end_str}'
""").to_pandas()
df_actuals.columns = df_actuals.columns.str.lower()
df_actuals['snapshot_date']    = pd.to_datetime(df_actuals['snapshot_date'])
df_actuals['principalbalance'] = df_actuals['principalbalance'].astype(float)
df_actuals['co_month']         = df_actuals['snapshot_date'].dt.to_period('M').astype(str)
actuals_by_month = df_actuals.groupby('co_month').agg(actual_co=('principalbalance','sum')).reset_index()

# ── 6. Pre-index calibration snapshots ────────────────────────────────────────
calib_snaps = {}
for snap_date, grp in df_calib.groupby('snapshot_date'):
    calib_snaps[pd.Timestamp(snap_date)] = (
        grp.sort_values('origday', ascending=False)
           .drop_duplicates(subset='advanceid', keep='first').copy()
    )

def get_raw_bal(snap_date, co_start, co_end):
    grp = calib_snaps.get(pd.Timestamp(snap_date))
    if grp is None or len(grp) == 0: return 0.0
    mask = (grp['chargeoff_date'] >= co_start) & (grp['chargeoff_date'] <= co_end)
    return grp.loc[mask, 'principalbalance'].sum()

def opt_rate(items):
    if not items: return np.nan
    total_raw = sum(x['raw'] for x in items)
    total_act = sum(x['actual'] for x in items)
    return (total_act / total_raw) if total_raw > 0 else np.nan

# ── 7. Compute flow rates ──────────────────────────────────────────────────────
imp = {1: [], 2: [], 3: []}
for c in calib_configs:
    act_row = actuals_by_month[actuals_by_month['co_month'] == c['co_month']]
    actual  = float(act_row['actual_co'].values[0]) if len(act_row) > 0 else 0.0
    for k, snap_key in [(3,'run_3mo'),(2,'run_2mo'),(1,'run_1mo')]:
        raw = get_raw_bal(c[snap_key], c['co_start'], c['co_end'])
        if raw > 0:
            imp[k].append({'raw': raw, 'actual': actual})
fr = {0: 1.0, 1: opt_rate(imp[1]), 2: opt_rate(imp[2]), 3: opt_rate(imp[3])}
print(f"Flow rates — M+1: {fr[1]:.4f}  M+2: {fr[2]:.4f}  M+3: {fr[3]:.4f}")

# ── 8. Live snapshot ──────────────────────────────────────────────────────────
df_live = session.sql(f"""
    select d.advanceid, d.origday,
        cast(d.date as date) as snapshot_date,
        dateadd(day, 120 - d.origday, cast(d.date as date)) as chargeoff_date,
        d.principalbalance
    from creditgenie.public.dailybalance d
    left join creditgenie.ddb_prod.advance a on d.advanceid = a.id
    where d.date = '{RUN_DATE.strftime('%Y-%m-%d')}'
      and d.origday < 120 and d.principalbalance > 0
      and a.status not in ('Failed','Cancelled','Canceled')
""").to_pandas()
df_live.columns = df_live.columns.str.lower()
df_live['chargeoff_date']    = pd.to_datetime(df_live['chargeoff_date'])
df_live['principalbalance']  = df_live['principalbalance'].astype(float)
df_live['origday']           = df_live['origday'].astype(int)
live = df_live.sort_values('origday', ascending=False).drop_duplicates(subset='advanceid', keep='first')
print(f'Live snapshot rows: {len(live):,}')

# ── 9. MTD actuals ────────────────────────────────────────────────────────────
month_start = RUN_DATE.to_period('M').to_timestamp().strftime('%Y-%m-%d')
df_mtd = session.sql(f"""
    select d.advanceid, d.principalbalance as chargeoff_amount, cast(d.date as date) as co_date
    from creditgenie.public.dailybalance d
    left join creditgenie.ddb_prod.advance a on d.advanceid = a.id
    where a.status not in ('Failed','Cancelled','Canceled')
      and d.origday = 120 and d.principalbalance > 0
      and cast(d.date as date) >= '{month_start}'
      and cast(d.date as date) <= '{RUN_DATE.strftime('%Y-%m-%d')}'
""").to_pandas()
df_mtd.columns = df_mtd.columns.str.lower()
df_mtd['chargeoff_amount'] = df_mtd['chargeoff_amount'].astype(float)
mtd_co = df_mtd['chargeoff_amount'].sum()

# ── 10. Forward forecast ──────────────────────────────────────────────────────
co_start_cur = RUN_DATE.to_period('M').to_timestamp()
MONTH_LABELS = {0: 'Current Month', 1: 'Next Month', 2: 'Month +2', 3: 'Month +3'}
proj = {}
orig_stats = {}
for i in range(4):
    co_s = co_start_cur + pd.DateOffset(months=i)
    co_e = co_s + pd.offsets.MonthEnd(0)
    mask = (live['chargeoff_date'] >= co_s) & (live['chargeoff_date'] <= co_e)
    sub  = live.loc[mask]
    raw  = sub['principalbalance'].sum() if len(sub) > 0 else 0.0
    rate = fr.get(i, np.nan)
    proj[i] = raw * rate if not np.isnan(rate) else np.nan
    if len(sub) > 0:
        orig_stats[i] = {
            'min': int(sub['origday'].min()),
            'max': int(sub['origday'].max()),
            'avg': round(float(sub['origday'].mean()), 1),
            'label': f"{co_s.strftime('%b %Y')}",
        }
    else:
        orig_stats[i] = {'min': None, 'max': None, 'avg': None, 'label': co_s.strftime('%b %Y')}

cur_rem  = proj.get(0, 0) or 0
cur_tot  = mtd_co + cur_rem
grand    = cur_tot + sum(proj.get(i, 0) or 0 for i in range(1, 4))
cur_month_label = co_start_cur.strftime('%B %Y')

# ── 11. Generate HTML ─────────────────────────────────────────────────────────
def fmt_usd(v):
    if v is None or (isinstance(v, float) and np.isnan(v)): return 'N/A'
    return f'${v:,.0f}'
def fmt_pct(v):
    if v is None or (isinstance(v, float) and np.isnan(v)): return 'N/A'
    return f'{v:.2%}'
def orig_combo(s):
    if s['min'] is None: return 'N/A'
    return f"{s['min']} – {s['avg']:.1f} – {s['max']}"

rows_forecast = [
    ('Flow Rate M+1', fmt_pct(fr[1]), ''),
    ('Flow Rate M+2', fmt_pct(fr[2]), ''),
    ('Flow Rate M+3', fmt_pct(fr[3]), ''),
    (f'{cur_month_label} — MTD Actual',  fmt_usd(mtd_co),  ''),
    (f'{cur_month_label} — Projected Remaining', fmt_usd(cur_rem), ''),
    (f'{cur_month_label} — TOTAL', fmt_usd(cur_tot), 'subtotal'),
]
for i in range(1, 4):
    co_s = co_start_cur + pd.DateOffset(months=i)
    rows_forecast.append((f'{co_s.strftime("%B %Y")} — Projected', fmt_usd(proj.get(i)), ''))
rows_forecast.append(('Grand Total (4 months)', fmt_usd(grand), 'grand'))

rows_orig = []
for i in range(4):
    s = orig_stats[i]
    rows_orig.append((s['label'], orig_combo(s)))

forecast_rows_html = ''
for label, value, cls in rows_forecast:
    row_cls = f' class="{cls}"' if cls else ''
    forecast_rows_html += f'<tr{row_cls}><td>{label}</td><td>{value}</td></tr>\n'

orig_rows_html = ''
for label, value in rows_orig:
    orig_rows_html += f'<tr><td>{label}</td><td>{value}</td></tr>\n'

html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Chargeoff Daily Forecast — {RUN_DATE.strftime('%B %d, %Y')}</title>
<style>
  body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
         background: #f4f6f9; margin: 0; padding: 24px; color: #1a1a2e; }}
  h1   {{ color: #1F4E79; margin-bottom: 4px; }}
  .subtitle {{ color: #666; font-size: 0.9em; margin-bottom: 28px; }}
  .grid {{ display: grid; grid-template-columns: 1fr 1fr; gap: 24px; max-width: 900px; }}
  .card {{ background: white; border-radius: 10px; padding: 20px;
           box-shadow: 0 2px 8px rgba(0,0,0,0.08); }}
  .card h2 {{ font-size: 1em; color: #1F4E79; margin: 0 0 14px; text-transform: uppercase;
              letter-spacing: 0.05em; border-bottom: 2px solid #DCE6F1; padding-bottom: 8px; }}
  table {{ width: 100%; border-collapse: collapse; font-size: 0.92em; }}
  th    {{ background: #1F4E79; color: white; padding: 8px 12px; text-align: left; }}
  td    {{ padding: 7px 12px; border-bottom: 1px solid #eef0f4; }}
  tr:nth-child(even) td {{ background: #f8fafc; }}
  tr.subtotal td {{ background: #DCE6F1 !important; font-weight: 600; }}
  tr.grand    td {{ background: #1F4E79 !important; color: white; font-weight: 700; font-size: 1.02em; }}
  .meta {{ max-width: 900px; margin-top: 16px; font-size: 0.8em; color: #999; }}
</style>
</head>
<body>
<h1>Chargeoff Daily Forecast</h1>
<div class="subtitle">Run Date: <strong>{RUN_DATE.strftime('%A, %B %d, %Y')}</strong>
  &nbsp;|&nbsp; Calibration window: Oct 2025 – Mar 2026 (6 months)
  &nbsp;|&nbsp; Day-of-month used for calibration: <strong>{run_day}</strong></div>

<div class="grid">
  <div class="card">
    <h2>Forecast Summary</h2>
    <table>
      <tr><th>Metric</th><th>Value</th></tr>
      {forecast_rows_html}
    </table>
  </div>
  <div class="card">
    <h2>Origday Range (min – avg – max)</h2>
    <table>
      <tr><th>Month</th><th>Origday as of Run Date</th></tr>
      {orig_rows_html}
    </table>
    <p style="font-size:0.78em;color:#999;margin-top:10px;">
      origday = days elapsed since advance origination (120 = chargeoff)
    </p>
  </div>
</div>

<div class="meta">
  Generated by GitHub Actions &nbsp;|&nbsp;
  Calibration: empirical OPTIMAL M+X rates (sum actual / sum raw)
</div>
</body>
</html>
"""

try:
    out_dir = Path('docs')
    out_dir.mkdir(exist_ok=True)
    (out_dir / 'index.html').write_text(html)
    print(f'✓ Report written to docs/index.html')

    # Also save a JSON snapshot for history
    history_file = out_dir / 'history.json'
    history = json.loads(history_file.read_text()) if history_file.exists() else []
    history.append({
        'run_date':   RUN_DATE.strftime('%Y-%m-%d'),
        'fr_m1':      round(fr[1], 4),
        'fr_m2':      round(fr[2], 4),
        'fr_m3':      round(fr[3], 4),
        'mtd_actual': round(mtd_co, 0),
        'cur_proj':   round(cur_rem, 0),
        'cur_total':  round(cur_tot, 0),
        'grand_total':round(grand, 0),
    })
    history_file.write_text(json.dumps(history, indent=2))
    print(f'✓ History updated ({len(history)} entries).')
    print('\n✓ Forecast complete!')
except Exception as e:
    print(f'✗ Failed to write report: {e}', file=sys.stderr)
    sys.exit(1)
finally:
    try:
        session.close()
    except:
        pass
