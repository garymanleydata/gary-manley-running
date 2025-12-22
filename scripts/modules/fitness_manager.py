import pandas as pd
import plotly.graph_objects as go
import numpy as np

def calculate_trimp(duration_min, avg_hr, max_hr, rest_hr):
    try:
        hrr = (avg_hr - rest_hr) / (max_hr - rest_hr)
        # Banister's TRIMP formula for men (generic factor 1.92)
        load = duration_min * hrr * 0.64 * np.exp(1.92 * hrr)
        return int(load)
    except:
        return 0

def parse_intervals_csv(file_obj):
    try:
        df = pd.read_csv(file_obj)
        # Map Intervals.icu columns to our schema
        # Required: Date, Activity Name, Type, Training Load, Duration, Zone times
        cols_map = {
            'start_date_local': 'Date',
            'name': 'Activity',
            'type': 'Type',
            'training_load': 'Load',
            'elapsed_time': 'Duration',
            'z1_time': 'Z1', 'z2_time': 'Z2', 'z3_time': 'Z3', 'z4_time': 'Z4', 'z5_time': 'Z5'
        }
        
        # Check if basic columns exist
        if not {'start_date_local', 'training_load'}.issubset(df.columns):
            return None, "Invalid Intervals CSV format"

        df = df.rename(columns=cols_map)
        
        # Clean Duration (seconds to minutes)
        if 'Duration' in df.columns:
            df['Duration'] = (df['Duration'] / 60).astype(int)
            
        # Clean Zones (seconds to minutes)
        for z in ['Z1', 'Z2', 'Z3', 'Z4', 'Z5']:
            if z in df.columns:
                df[z] = (df[z] / 60).fillna(0).astype(int)
            else:
                df[z] = 0
                
        # Filter Columns
        keep_cols = ['Date', 'Activity', 'Type', 'Load', 'Duration', 'Z1', 'Z2', 'Z3', 'Z4', 'Z5']
        # Fill missing with defaults
        for c in keep_cols:
            if c not in df.columns: df[c] = 0
            
        return df[keep_cols], "Success"
    except Exception as e:
        return None, str(e)

def parse_strava_csv(file_obj, user_max_hr, user_rest_hr):
    try:
        df = pd.read_csv(file_obj)
        
        # Strava export is often messy. We look for: 'Activity Date', 'Activity Type', 'Elapsed Time', 'Max Heart Rate'
        # Note: Strava CSV export usually doesn't have Load or Avg HR easily. 
        # This is a basic parser assuming standard Bulk Export.
        
        cols_map = {
            'Activity Date': 'Date',
            'Activity Name': 'Activity',
            'Activity Type': 'Type',
            'Elapsed Time': 'Duration',
            'Average Heart Rate': 'AvgHR' # Might not exist in all exports
        }
        
        if 'Activity Date' not in df.columns:
            return None, "Invalid Strava CSV"

        df = df.rename(columns=cols_map)
        
        # Clean Date
        df['Date'] = pd.to_datetime(df['Date']).dt.date
        
        # Clean Duration (seconds to minutes)
        df['Duration'] = (df['Duration'] / 60).astype(int)
        
        # Calculate Load if missing (requires Avg HR)
        if 'AvgHR' in df.columns:
            df['Load'] = df.apply(lambda x: calculate_trimp(x['Duration'], pd.to_numeric(x['AvgHR'], errors='coerce'), user_max_hr, user_rest_hr), axis=1)
        else:
            df['Load'] = df['Duration'] * 0.5 # Dummy load if no HR
            
        # Zones are usually missing in Strava CSV, set to 0
        for z in ['Z1', 'Z2', 'Z3', 'Z4', 'Z5']:
            df[z] = 0
            
        keep_cols = ['Date', 'Activity', 'Type', 'Load', 'Duration', 'Z1', 'Z2', 'Z3', 'Z4', 'Z5']
        for c in keep_cols:
            if c not in df.columns: df[c] = 0
            
        return df[keep_cols], "Success"
    except Exception as e:
        return None, str(e)

def calculate_pmc(df_history):
    # Ensure sorted by date
    df = df_history.sort_values('Date').copy()
    df['Date'] = pd.to_datetime(df['Date'])
    df = df.set_index('Date')
    
    # Resample to daily to handle missing days (fill with 0 load)
    df_daily = df.resample('D')['Load'].sum().to_frame()
    
    # Calculate Rolling Averages
    # CTL = Fitness (42 day exp avg)
    # ATL = Fatigue (7 day exp avg)
    df_daily['CTL'] = df_daily['Load'].ewm(span=42, adjust=False).mean()
    df_daily['ATL'] = df_daily['Load'].ewm(span=7, adjust=False).mean()
    
    # TSB = Form (CTL - ATL) NOTE: Usually TSB = Previous Day CTL - Previous Day ATL
    # But standard simplified is Fitness - Fatigue
    df_daily['TSB'] = df_daily['CTL'] - df_daily['ATL']
    
    return df_daily

def create_pmc_figure(df_pmc):
    fig = go.Figure()
    
    # Form (TSB) - Area / Bar
    fig.add_trace(go.Bar(
        x=df_pmc.index, 
        y=df_pmc['TSB'], 
        name='Form (TSB)',
        marker_color='gray',
        opacity=0.3
    ))
    
    # Fatigue (ATL) - Dotted Line
    fig.add_trace(go.Scatter(
        x=df_pmc.index, 
        y=df_pmc['ATL'], 
        name='Fatigue (ATL)',
        line=dict(color='#ff7f0e', dash='dot')
    ))
    
    # Fitness (CTL) - Solid Line (Thick)
    fig.add_trace(go.Scatter(
        x=df_pmc.index, 
        y=df_pmc['CTL'], 
        name='Fitness (CTL)',
        line=dict(color='#1f77b4', width=3),
        fill='tozeroy',
        fillcolor='rgba(31, 119, 180, 0.1)'
    ))
    
    fig.update_layout(
        title="Performance Management Chart (PMC)",
        xaxis_title="Date",
        yaxis_title="Load (TSS)",
        hovermode="x unified",
        legend=dict(orientation="h", y=1.1),
        template="plotly_white",
        height=400
    )
    return fig

def create_zone_figure(df_history):
    # Prepare Data
    df = df_history.copy()
    df['Date'] = pd.to_datetime(df['Date'])
    
    # --- FIX: Group by Week Starting MONDAY ---
    # W-MON means "Weekly frequency, anchored to Mondays"
    # closed='left', label='left' ensures the label is the Monday date
    df_weekly = df.resample('W-MON', on='Date', closed='left', label='left')[['Z1', 'Z2', 'Z3', 'Z4', 'Z5']].sum()
    
    # Reformat index for display
    df_weekly.index = df_weekly.index.strftime('%Y-%m-%d')
    
    fig = go.Figure()
    zones = ['Z1', 'Z2', 'Z3', 'Z4', 'Z5']
    colors = ['#a6cee3', '#1f78b4', '#b2df8a', '#33a02c', '#e31a1c']
    
    for i, z in enumerate(zones):
        fig.add_trace(go.Bar(
            x=df_weekly.index, 
            y=df_weekly[z], 
            name=z,
            marker_color=colors[i]
        ))
        
    fig.update_layout(
        title="Weekly Zone Distribution (Minutes)",
        barmode='stack',
        xaxis_title="Week Commencing (Monday)",
        yaxis_title="Minutes",
        template="plotly_white",
        height=400
    )
    return fig