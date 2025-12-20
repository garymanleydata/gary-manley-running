import pandas as pd
import numpy as np
import plotly.graph_objects as go
from datetime import datetime, timedelta

def calculate_trimp(duration_min, avg_hr, max_hr, rest_hr):
    """Calculates TRIMP (Training Impulse) based on Banister's formula."""
    if not avg_hr or not max_hr or not rest_hr: return 0
    hr_reserve = (avg_hr - rest_hr) / (max_hr - rest_hr)
    trimp = duration_min * hr_reserve * 0.64 * np.exp(1.92 * hr_reserve)
    return int(trimp)

def parse_intervals_csv(file_obj):
    """Parses Intervals.icu CSV export for Load and Zones."""
    try:
        df = pd.read_csv(file_obj)
        
        # Mapping columns (Intervals.icu variable naming)
        col_map = {
            'icu_training_load': 'Load',
            'training_load': 'Load',
            'moving_time': 'Duration',
            'start_date': 'Date',
            'start_time_local': 'Date', # Sometimes date is here
            'name': 'Activity',
            'type': 'Type'
        }
        
        # Rename known columns
        df.rename(columns=col_map, inplace=True)
        
        # Required columns check
        if 'Load' not in df.columns or 'Date' not in df.columns:
            return None, "Missing 'Load' or 'Date' columns."
            
        # Extract Zones if available (z1_time, z2_time OR zone1_time)
        # We normalize to minutes (Intervals usually gives seconds)
        for i in range(1, 6):
            z_col = f'Z{i}'
            # Try to find matching column in input
            possible_names = [f'z{i}_time', f'zone{i}_time', f'z{i}_moving_time']
            found_col = next((c for c in df.columns if c in possible_names), None)
            
            if found_col:
                df[z_col] = df[found_col] / 60 # Convert seconds to mins
            else:
                df[z_col] = 0
                
        # Final cleanup
        keep_cols = ['Date', 'Activity', 'Type', 'Load', 'Duration', 'Z1', 'Z2', 'Z3', 'Z4', 'Z5']
        # Ensure all exist
        for c in keep_cols:
            if c not in df.columns: df[c] = 0
            
        df = df[keep_cols].copy()
        df['Date'] = pd.to_datetime(df['Date']).dt.date
        df['Load'] = df['Load'].fillna(0).astype(int)
        df['Duration'] = (df['Duration'] / 60).astype(int) # Sec to Min
        
        return df, "Success"
    except Exception as e:
        return None, f"Error: {e}"

def parse_strava_csv(file_obj, max_hr, rest_hr):
    """Parses Strava CSV and CALCULATES Load. No Zones available."""
    try:
        df = pd.read_csv(file_obj)
        req_cols = ['Activity Date', 'Activity Name', 'Activity Type', 'Moving Time', 'Average Heart Rate']
        if not all(col in df.columns for col in req_cols):
             return None, "Strava CSV format mismatch."

        data = []
        for _, row in df.iterrows():
            try:
                date = pd.to_datetime(row['Activity Date']).date()
                duration_min = row['Moving Time'] / 60
                avg_hr = pd.to_numeric(row['Average Heart Rate'], errors='coerce')
                
                load = 0
                if pd.notna(avg_hr) and avg_hr > 0:
                    load = calculate_trimp(duration_min, avg_hr, max_hr, rest_hr)
                
                # Strava doesn't give zone data in bulk export
                # We assume 0 for now (User must manually edit if they care)
                data.append({
                    'Date': date, 'Activity': row['Activity Name'], 'Type': row['Activity Type'],
                    'Load': load, 'Duration': int(duration_min),
                    'Z1': 0, 'Z2': 0, 'Z3': 0, 'Z4': 0, 'Z5': 0
                })
            except: continue
            
        return pd.DataFrame(data), "Success"
    except Exception as e:
        return None, f"Error: {e}"

def calculate_pmc(df_history):
    """Calculates CTL, ATL, TSB."""
    df_daily = df_history.groupby('Date')['Load'].sum().reset_index()
    df_daily = df_daily.sort_values('Date')
    
    full_idx = pd.date_range(start=df_daily['Date'].min(), end=df_daily['Date'].max())
    df_daily = df_daily.set_index('Date').reindex(full_idx.date).fillna(0).reset_index()
    df_daily.rename(columns={'index': 'Date'}, inplace=True)
    
    df_daily['CTL'] = df_daily['Load'].ewm(span=42, adjust=False).mean()
    df_daily['ATL'] = df_daily['Load'].ewm(span=7, adjust=False).mean()
    df_daily['TSB'] = df_daily['CTL'] - df_daily['ATL']
    return df_daily

def create_pmc_figure(df_pmc):
    fig = go.Figure()
    colors = ['#2ca02c' if v >= -10 and v <= 10 else '#d62728' if v < -30 else '#ff7f0e' for v in df_pmc['TSB']]
    fig.add_trace(go.Bar(x=df_pmc['Date'], y=df_pmc['TSB'], name='Form (TSB)', marker_color=colors, opacity=0.3))
    fig.add_trace(go.Scatter(x=df_pmc['Date'], y=df_pmc['CTL'], name='Fitness (CTL)', fill='tozeroy', line=dict(color='#1f77b4', width=2)))
    fig.add_trace(go.Scatter(x=df_pmc['Date'], y=df_pmc['ATL'], name='Fatigue (ATL)', line=dict(color='#e377c2', width=1, dash='dot')))
    fig.update_layout(title="Fitness Trend (CTL)", hovermode="x unified", height=400, yaxis_title="Score", margin=dict(l=20, r=20, t=40, b=20))
    return fig

def create_zone_figure(df_history):
    """Creates Weekly Intensity Distribution Chart."""
    # Ensure date format
    df = df_history.copy()
    df['Date'] = pd.to_datetime(df['Date'])
    
    # Resample to Weekly sums
    # We select only the zone columns
    z_cols = ['Z1', 'Z2', 'Z3', 'Z4', 'Z5']
    df_weekly = df.set_index('Date')[z_cols].resample('W-MON').sum().reset_index()
    
    fig = go.Figure()
    colors = ['#a6cee3', '#1f78b4', '#b2df8a', '#33a02c', '#e31a1c']
    labels = ['Z1 (Recov)', 'Z2 (Aerobic)', 'Z3 (Tempo)', 'Z4 (Thresh)', 'Z5 (VO2)']
    
    for i, col in enumerate(z_cols):
        fig.add_trace(go.Bar(x=df_weekly['Date'], y=df_weekly[col], name=labels[i], marker_color=colors[i]))
        
    fig.update_layout(
        title="Weekly Intensity Distribution", 
        barmode='stack', 
        hovermode="x unified", 
        height=400, 
        yaxis_title="Minutes",
        margin=dict(l=20, r=20, t=40, b=20)
    )
    return fig