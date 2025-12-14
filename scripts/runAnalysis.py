print("--- Script Initializing ---")
import gpxpy
import pandas as pd
import folium
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import os
import numpy as np
from tcxreader.tcxreader import TCXReader
from datetime import datetime
import sys

# 1. Setup Configuration
# ---------------------------------------------------------
vScriptDir = os.path.dirname(os.path.abspath(__file__))

# === USER SETTINGS ===
vRunFilename = 'Gary_Manley_2025-12-13.tcx'  # <--- The run you did TODAY
vGhostFilename = 'Gary_Manley_2025-12-06.tcx'                         # <--- (Optional) 'Gary_PB.tcx'
vMaxHR = 185                                  # <--- Your Max HR
# =====================

# Paths
vInputPath = os.path.join(vScriptDir, f'../data/runs/{vRunFilename}')
vImagesPath = os.path.join(vScriptDir, '../assets/images')
vGraphsPath = os.path.join(vScriptDir, '../assets/graphs') # NEW: For interactive HTML graphs
vMapPath = os.path.join(vScriptDir, '../assets/maps')
vTablesPath = os.path.join(vScriptDir, '../assets/tables')
vPostsPath = os.path.join(vScriptDir, '../posts')

for path in [vImagesPath, vGraphsPath, vMapPath, vTablesPath, vPostsPath]:
    os.makedirs(path, exist_ok=True)

# 2. Universal Parser
# ---------------------------------------------------------
def parse_run_file(vFilename):
    vFilePath = os.path.join(vScriptDir, f'../data/runs/{vFilename}')
    if not os.path.exists(vFilePath):
        print(f"ERROR: File not found: {vFilePath}")
        sys.exit(1)

    print(f"  > Parsing: {vFilename}...")
    vExt = os.path.splitext(vFilePath)[1].lower()
    data = []

    try:
        if vExt == '.gpx':
            with open(vFilePath, 'r') as gpx_file:
                gpx = gpxpy.parse(gpx_file)
            for track in gpx.tracks:
                for segment in track.segments:
                    for point in segment.points:
                        hr, cad = None, None
                        if point.extensions:
                            for ext in point.extensions:
                                if 'TrackPointExtension' in ext.tag:
                                    for child in ext:
                                        if 'hr' in child.tag: hr = int(child.text)
                                        if 'cad' in child.tag: cad = int(child.text)
                        data.append({'time': point.time, 'lat': point.latitude, 'lon': point.longitude, 'ele': point.elevation, 'hr': hr, 'cadence': cad})

        elif vExt == '.tcx':
            tcx_reader = TCXReader()
            activity = tcx_reader.read(vFilePath)
            for tp in activity.trackpoints:
                data.append({'time': tp.time, 'lat': tp.latitude, 'lon': tp.longitude, 'ele': tp.elevation, 'hr': tp.hr_value, 'cadence': tp.cadence})

        df = pd.DataFrame(data)
        
        # Calculations
        df['prev_lat'] = df['lat'].shift(1)
        df['prev_lon'] = df['lon'].shift(1)
        df['prev_time'] = df['time'].shift(1)
        
        R = 6371000 
        phi1 = np.radians(df['lat'])
        phi2 = np.radians(df['prev_lat'])
        dphi = np.radians(df['prev_lat'] - df['lat'])
        dlambda = np.radians(df['prev_lon'] - df['lon'])
        a = np.sin(dphi/2)**2 + np.cos(phi1)*np.cos(phi2) * np.sin(dlambda/2)**2
        c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1-a))
        df['dist_m'] = R * c
        df['dist_m'] = df['dist_m'].fillna(0)
        df['total_dist_m'] = df['dist_m'].cumsum()
        df['time_diff'] = (df['time'] - df['prev_time']).dt.total_seconds()
        
        df = df[df['time_diff'] > 0]
        df['speed_mps'] = df['dist_m'] / df['time_diff']
        
        # Smart Cadence & Stride
        df['cadence'] = df['cadence'].fillna(0)
        max_cad = df['cadence'].max()
        if max_cad > 0:
            if max_cad < 130:
                df['cadence'] = df['cadence'] * 2
            df['stride_len'] = np.where((df['cadence'] > 60) & (df['speed_mps'] > 0.5), df['speed_mps'] / (df['cadence'] / 60), np.nan)
        else:
            df['stride_len'] = 0

        return df
    except Exception as e:
        print(f"CRITICAL ERROR parsing {vFilename}: {e}")
        sys.exit(1)

# 3. Create Map (Folium)
# ---------------------------------------------------------
def create_map(df, vDateStr):
    try:
        vStartLat = df.iloc[0]['lat']
        vStartLon = df.iloc[0]['lon']
        vMap = folium.Map(location=[vStartLat, vStartLon], zoom_start=14, tiles='CartoDB positron')
        vAvgSpeed = df['speed_mps'].mean()
        vColormap = plt.get_cmap('RdYlGn') 
        df_map = df.iloc[::2]
        for i in range(1, len(df_map)):
            vSpeed = df_map.iloc[i]['speed_mps']
            vNorm = max(0, min(1, (vSpeed - (vAvgSpeed * 0.5)) / (vAvgSpeed)))
            folium.PolyLine(
                locations=[(df_map.iloc[i-1]['lat'], df_map.iloc[i-1]['lon']), (df_map.iloc[i]['lat'], df_map.iloc[i]['lon'])],
                color=mcolors.to_hex(vColormap(vNorm)), weight=5, opacity=0.8
            ).add_to(vMap)
        vMapName = f"map_{vDateStr}.html"
        vMap.save(os.path.join(vMapPath, vMapName))
        print(f"  > Map saved: {vMapName}")
    except Exception as e:
        print(f"  ! Error creating map: {e}")

# 4. Generate Table
# ---------------------------------------------------------
def create_segment_table(df, vDateStr):
    has_hr = df['hr'].sum() > 0
    has_cad = df['cadence'].sum() > 0
    has_stride = 'stride_len' in df.columns and df['stride_len'].mean() > 0

    df['segment_100m'] = (df['total_dist_m'] // 100).astype(int) + 1
    
    agg_dict = {'time_diff': 'sum', 'dist_m': 'sum'}
    if has_hr: agg_dict['hr'] = 'mean'
    if has_cad: agg_dict['cadence'] = 'mean'
    if has_stride: agg_dict['stride_len'] = 'mean'

    df_agg = df.groupby('segment_100m').agg(agg_dict).reset_index()
    df_agg = df_agg[df_agg['dist_m'] > 50]
    
    df_agg['split_time'] = df_agg['time_diff'].apply(lambda x: f"{int(x)//60}:{int(x)%60:02d}")
    df_agg['pace_decimal'] = (df_agg['time_diff'] / df_agg['dist_m']) * (1000/60)
    df_agg['pace_str'] = df_agg['pace_decimal'].apply(lambda x: f"{int(x)}:{int((x-int(x))*60):02d}")
    
    cols = ['segment_100m', 'split_time', 'pace_str']
    headers = ['Dist (100m)', 'Time', 'Pace']
    
    if has_hr:
        df_agg['avg_hr'] = pd.to_numeric(df_agg['hr'], errors='coerce').fillna(0).astype(int)
        cols.append('avg_hr'); headers.append('HR')
    if has_cad:
        df_agg['avg_cad'] = pd.to_numeric(df_agg['cadence'], errors='coerce').fillna(0).astype(int)
        cols.append('avg_cad'); headers.append('Cadence')
    if has_stride:
        df_agg['avg_stride'] = pd.to_numeric(df_agg['stride_len'], errors='coerce').fillna(0).round(2)
        cols.append('avg_stride'); headers.append('Stride (m)')

    vTableDF = df_agg[cols]
    vTableDF.columns = headers
    vTableName = f"table_{vDateStr}.md"
    vTableDF.to_markdown(os.path.join(vTablesPath, vTableName), index=False)
    print(f"  > Table saved: {vTableName}")

# 5. NEW: Interactive Analysis Dashboard (Plotly)
# ---------------------------------------------------------
def create_interactive_dashboard(df, vDateStr):
    # Aggregate data
    df['segment_100m'] = (df['total_dist_m'] // 100).astype(int) + 1
    agg_dict = {'time_diff': 'sum'}
    if 'stride_len' in df.columns: agg_dict['stride_len'] = 'mean'
    if 'cadence' in df.columns: agg_dict['cadence'] = 'mean'
    if 'hr' in df.columns: agg_dict['hr'] = 'mean'
    
    df_agg = df.groupby('segment_100m').agg(agg_dict).reset_index()
    df_agg = df_agg[df_agg['time_diff'] > 0]
    
    # Calculate Variance
    avg_time = df_agg['time_diff'].mean()
    df_agg['variance'] = df_agg['time_diff'] - avg_time
    # Color logic: Green = Fast (Negative variance), Red = Slow (Positive variance)
    colors = ['#2ca02c' if v <= 0 else '#d62728' for v in df_agg['variance']]

    # Create Subplots
    fig = make_subplots(
        rows=2, cols=1, 
        shared_xaxes=True, 
        vertical_spacing=0.1,
        subplot_titles=("Pacing Discipline (vs Average)", "Mechanics: Stride & Cadence"),
        specs=[[{"secondary_y": False}], [{"secondary_y": True}]]
    )

    # 1. Pace Variance Bar Chart
    fig.add_trace(go.Bar(
        x=df_agg['segment_100m'],
        y=df_agg['variance'],
        marker_color=colors,
        name='Pace Variance (s)',
        hovertemplate='Segment: %{x}<br>Variance: %{y:.1f}s<extra></extra>'
    ), row=1, col=1)

    # 2. Stride Length (Left Axis)
    if 'stride_len' in df_agg.columns and df_agg['stride_len'].mean() > 0:
        fig.add_trace(go.Scatter(
            x=df_agg['segment_100m'],
            y=df_agg['stride_len'],
            mode='lines',
            name='Stride Length (m)',
            line=dict(color='blue', width=3),
            hovertemplate='Stride: %{y:.2f}m<extra></extra>'
        ), row=2, col=1, secondary_y=False)

    # 3. Cadence (Right Axis)
    if 'cadence' in df_agg.columns and df_agg['cadence'].mean() > 0:
        fig.add_trace(go.Scatter(
            x=df_agg['segment_100m'],
            y=df_agg['cadence'],
            mode='lines',
            name='Cadence (spm)',
            line=dict(color='orange', width=2, dash='dot'),
            hovertemplate='Cadence: %{y:.0f} spm<extra></extra>'
        ), row=2, col=1, secondary_y=True)

    # Layout Updates
    fig.update_layout(height=700, title_text=f"Run Analysis Dashboard: {vDateStr}", hovermode="x unified")
    fig.update_yaxes(title_text="Seconds (+ Slower / - Faster)", row=1, col=1)
    fig.update_yaxes(title_text="Stride Length (m)", row=2, col=1, secondary_y=False)
    fig.update_yaxes(title_text="Cadence (spm)", row=2, col=1, secondary_y=True)
    fig.update_xaxes(title_text="Distance (x100m)", row=2, col=1)

    # Save
    vHtmlName = f"dashboard_{vDateStr}.html"
    fig.write_html(os.path.join(vGraphsPath, vHtmlName))
    print(f"  > Interactive Dashboard saved: {vHtmlName}")
    
    # --- STATIC THUMBNAIL (For Blog Cover) ---
    # We create a simple Matplotlib version just for the cover image
    plt.figure(figsize=(10, 5))
    plt.bar(df_agg['segment_100m'], df_agg['time_diff'], color='#1f77b4', alpha=0.8)
    plt.title(f"Run Summary: {vDateStr}")
    plt.ylabel("Seconds")
    plt.xlabel("Segment #")
    vThumbName = f"thumb_{vDateStr}.png"
    plt.savefig(os.path.join(vImagesPath, vThumbName))
    plt.close()
    
    return vHtmlName, vThumbName

# 6. HR Zones (Calculated Only)
# ---------------------------------------------------------
def create_hr_analysis(df):
    if df['hr'].sum() == 0: return None
    
    weighted_score = 0
    zones = [(0.5,0.6,1),(0.6,0.7,2),(0.7,0.8,3),(0.8,0.9,4),(0.9,2.0,5)]
    
    for (low, high, score) in zones:
        mask = (df['hr'] >= (low*vMaxHR)) & (df['hr'] < (high*vMaxHR))
        minutes = df.loc[mask, 'time_diff'].sum() / 60
        weighted_score += score * minutes
        
    return int(weighted_score)

# 7. Interactive Ghost Battle
# ---------------------------------------------------------
def create_ghost_comparison(dfMain, dfGhost, vDateStr):
    print("  > Generating Interactive Ghost Battle...")
    
    max_dist = min(dfMain['total_dist_m'].max(), dfGhost['total_dist_m'].max())
    grid = np.arange(0, max_dist, 50)
    
    time_main = np.interp(grid, dfMain['total_dist_m'], dfMain['time'].astype(np.int64) // 10**9)
    time_ghost = np.interp(grid, dfGhost['total_dist_m'], dfGhost['time'].astype(np.int64) // 10**9)
    
    time_main -= time_main[0]
    time_ghost -= time_ghost[0]
    gap = time_main - time_ghost
    
    # Plotly Graph
    fig = go.Figure()
    
    # Main Gap Line
    fig.add_trace(go.Scatter(
        x=grid, y=gap,
        mode='lines',
        name='Gap to Ghost',
        line=dict(color='black', width=1),
        fill='tozeroy',
        fillcolor='rgba(0, 255, 0, 0.2)' # Default green
    ))
    
    # Add colored regions logic requires complex Plotly shapes, 
    # simpler is just one fill with conditional color which Plotly doesn't support easily on one trace.
    # Instead we rely on the line + hover to tell the story.
    
    fig.update_layout(
        title="Ghost Battle: Time Gap (Seconds)",
        xaxis_title="Distance (m)",
        yaxis_title="Gap (Seconds) [Neg=Ahead, Pos=Behind]",
        hovermode="x unified"
    )
    
    vHtmlName = f"battle_{vDateStr}.html"
    fig.write_html(os.path.join(vGraphsPath, vHtmlName))
    print(f"  > Battle Dashboard saved: {vHtmlName}")
    return vHtmlName

# 8. Create Blog Post
# ---------------------------------------------------------
def create_blog_post(vDateStr, hr_score, dashboard_file, thumb_file, battle_file=None):
    vRelMap = f"../assets/maps/map_{vDateStr}.html"
    vRelDash = f"../assets/graphs/{dashboard_file}"
    vRelThumb = f"../assets/images/{thumb_file}"
    vRelTable = f"../assets/tables/table_{vDateStr}.md"
    
    hr_text = f"\n**Effort Score:** {hr_score}\n" if hr_score else ""
    
    battle_section = ""
    if battle_file:
        battle_section = f"""
## The Ghost Battle
<iframe src="../assets/graphs/{battle_file}" width="100%" height="500px" style="border:none;"></iframe>
"""

    vPostFilename = f"{vDateStr}-run.qmd"
    vFullPath = os.path.join(vPostsPath, vPostFilename)
    
    vTemplate = f"""---
title: "Run Analysis: {vDateStr}"
date: "{vDateStr}"
categories: [running, analysis]
image: {vRelThumb}
format:
  html:
    toc: true
---

## Route Map
<iframe src="{vRelMap}" width="100%" height="500px" style="border:none;"></iframe>

{hr_text}

{battle_section}

## Performance Dashboard
Interactive: Hover to see details.
<iframe src="{vRelDash}" width="100%" height="700px" style="border:none;"></iframe>

## Detailed Splits
{{{{< include {vRelTable} >}}}}
"""
    with open(vFullPath, 'w') as f:
        f.write(vTemplate)
    print(f"\nSUCCESS: Blog post created at: {vFullPath}")

# Main Execution
if __name__ == "__main__":
    print("[1/6] Loading Main Run...")
    dfRun = parse_run_file(vRunFilename)
    vRunDate = dfRun['time'].iloc[0].strftime('%Y-%m-%d')
    print(f"      Detected Date: {vRunDate}")

    print("[2/6] Generating Map...")
    create_map(dfRun, vRunDate)
    
    print("[3/6] Generating Table...")
    create_segment_table(dfRun, vRunDate)
    
    print("[4/6] Generating Interactive Dashboards...")
    dash_html, thumb_png = create_interactive_dashboard(dfRun, vRunDate)
    
    print("[5/6] Calculating Effort...")
    hr_score = create_hr_analysis(dfRun)
    
    battle_html = None
    if vGhostFilename:
        print("[OPTIONAL] Ghost Runner Detected...")
        dfGhost = parse_run_file(vGhostFilename)
        battle_html = create_ghost_comparison(dfRun, dfGhost, vRunDate)

    print("[6/6] Writing Blog Post...")
    create_blog_post(vRunDate, hr_score, dash_html, thumb_png, battle_html)
    print("--- Done ---")