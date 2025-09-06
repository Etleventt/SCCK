"""
Interactive metrics explorer (fallback script when Jupyter is not available).
Run inside Jupyter (via %run) or as a plain script to pop up an interactive
widget panel if running in a notebook. Otherwise, customize variables at bottom
and call plot_runs.
"""
import os, glob, math
from pathlib import Path
import pandas as pd
import numpy as np
import plotly.graph_objects as go

pd.options.mode.copy_on_write = True

def find_csvs(patterns):
    if isinstance(patterns, str):
        patterns = [p.strip() for p in patterns.split(';') if p.strip()]
    files = []
    for pat in patterns:
        files += glob.glob(pat, recursive=True)
    files = [f for f in sorted(set(files)) if os.path.basename(f) == 'metrics.csv']
    runs = []
    for f in files:
        d = os.path.dirname(f)
        parts = Path(d).parts
        name = '/'.join(parts[-3:])
        runs.append((name, f))
    return runs

def available_columns(csv_path):
    with open(csv_path, 'r', encoding='utf-8') as fh:
        header = fh.readline().strip().split(',')
    return header

def load_metrics(csv_path, metrics, align_by='step', chunksize=20000):
    cols = []
    header = available_columns(csv_path)
    if 'epoch' in header: cols.append('epoch')
    if 'step' in header: cols.append('step')
    for m in metrics:
        if m in header: cols.append(m)
    cols = list(dict.fromkeys(cols))
    if not cols: return pd.DataFrame()
    usecols = cols
    dfs = []
    for chunk in pd.read_csv(csv_path, chunksize=chunksize, usecols=lambda c: c in usecols):
        mask_any = False
        for m in metrics:
            if m in chunk.columns:
                mm = chunk[m].notna()
                mask_any = mm if mask_any is False else (mask_any | mm)
        if mask_any is False:
            continue
        sub = chunk.loc[mask_any, usecols].copy()
        dfs.append(sub)
    if not dfs:
        return pd.DataFrame(columns=usecols)
    df = pd.concat(dfs, ignore_index=True)
    for c in ('epoch','step'):
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors='coerce')
    df.sort_values(by=[c for c in (align_by, 'epoch') if c in df.columns], inplace=True)
    return df

def ema_smooth(y, alpha=0.0):
    if alpha <= 0: return y
    out = np.zeros_like(y, dtype=float)
    acc = None
    for i, v in enumerate(y):
        acc = v if acc is None else (alpha * v + (1 - alpha) * acc)
        out[i] = acc
    return out

def plot_runs(run_paths, metrics, align_by='step', alpha=0.0, stride=1):
    fig = go.Figure()
    for name, path in run_paths:
        df = load_metrics(path, metrics, align_by=align_by)
        if df.empty: continue
        x = df[align_by] if align_by in df.columns else np.arange(len(df))
        for m in metrics:
            if m not in df.columns: continue
            y = df[m].to_numpy(dtype=float)
            if alpha>0: y = ema_smooth(y, alpha)
            if stride>1:
                x_ds = x.iloc[::stride] if hasattr(x, 'iloc') else x[::stride]
                y_ds = y[::stride]
            else:
                x_ds, y_ds = x, y
            fig.add_trace(go.Scatter(x=x_ds, y=y_ds, mode='lines', name=f"{name} — {m}"))
    fig.update_layout(template='plotly_white', xaxis_title=align_by, yaxis_title='metric value',
                      legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='left', x=0))
    return fig

if __name__ == '__main__':
    # Quick example: edit these lines and run `python notebooks/metrics_explorer.py`
    patterns = 'logs/**/version_*/metrics.csv'
    runs = find_csvs(patterns)
    metrics = ['val_psnr','val_ssim']
    fig = plot_runs(runs, metrics, align_by='step', alpha=0.0, stride=1)
    try:
        import plotly.io as pio
        pio.show(fig)
    except Exception:
        print('Figure created. Use fig.show() inside a notebook to render.')

