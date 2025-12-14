print("--- Script Initializing ---")
import gpxpy
import pandas as pd
import folium
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
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
vGhostFilename = None                         # <--- (Optional) The PB file to battle against
vMaxHR = 185                                  # <--- Your Max HR for Zone calc
# =====================

# Paths
vDataPath = os.path.join(vScriptDir, '../data/runs')
vImagesPath = os.path.join(vScriptDir, '../assets/images')
vMapPath = os.path.join(vScriptDir, '../assets/maps')
vTablesPath = os.path.join(vScriptDir, '../assets/tables')
vPostsPath = os.path.join(vScriptDir, '../posts')

for path in [vImagesPath, vMapPath, vTablesPath, vPostsPath]:
    os.makedirs(path, exist_ok=True)

# 2. Universal Parser
# ---------------------------------------------------------
def parse_run_file(vFilename):
    vFilePath = os.path.join(vDataPath, vFilename)
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
        
        # Filter valid movement
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

# 3. Create Map
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
    # FIX: Use simple existence check like graph
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
        # FIX: Ensure proper rounding and numeric conversion
        df_agg['avg_stride'] = pd.to_numeric(df_agg['stride_len'], errors='coerce').fillna(0).round(2)
        cols.append('avg_stride'); headers.append('Stride (m)')

    vTableDF = df_agg[cols]
    vTableDF.columns = headers
    vTableName = f"table_{vDateStr}.md"
    vTableDF.to_markdown(os.path.join(vTablesPath, vTableName), index=False)
    print(f"  > Table saved: {vTableName}")

# 5. Advanced Analysis Graph
# ---------------------------------------------------------
def create_analysis_graph(df, vDateStr):
    df['segment_100m'] = (df['total_dist_m'] // 100).astype(int) + 1
    agg_dict = {'time_diff': 'sum'}
    if 'stride_len' in df.columns: agg_dict['stride_len'] = 'mean'
    if 'cadence' in df.columns: agg_dict['cadence'] = 'mean'
    df_agg = df.groupby('segment_100m').agg(agg_dict).reset_index()
    df_agg = df_agg[df_agg['time_diff'] > 0]
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
    
    avg_seg_time = df_agg['time_diff'].mean()
    df_agg['variance'] = df_agg['time_diff'] - avg_seg_time
    colors = ['#d62728' if v > 0 else '#2ca02c' for v in df_agg['variance']]
    ax1.bar(df_agg['segment_100m'], df_agg['variance'], color=colors, alpha=0.8)
    ax1.axhline(0, color='black', linewidth=1)
    ax1.set_title("Pace Discipline (Seconds deviation from Average)", fontsize=12)
    ax1.set_ylabel("Seconds (+ Slower / - Faster)")

    if 'stride_len' in df_agg.columns and df_agg['stride_len'].mean() > 0:
        color = 'tab:blue'
        ax2.set_ylabel('Stride Length (m)', color=color)
        ax2.plot(df_agg['segment_100m'], df_agg['stride_len'], color=color, linewidth=2)
        ax2.tick_params(axis='y', labelcolor=color)
        ax3 = ax2.twinx()
        color = 'tab:orange'
        ax3.set_ylabel('Cadence (spm)', color=color)
        ax3.plot(df_agg['segment_100m'], df_agg['cadence'], color=color, linestyle='--', linewidth=2)
        ax3.tick_params(axis='y', labelcolor=color)
    else:
        ax2.text(0.5, 0.5, "No Cadence/Stride Data", ha='center', va='center')
    
    plt.tight_layout()
    vImgName = f"analysis_{vDateStr}.png"
    plt.savefig(os.path.join(vImagesPath, vImgName))
    plt.close()
    print(f"  > Analysis graph saved: {vImgName}")

# 6. HR Zones
# ---------------------------------------------------------
def create_hr_analysis(df, vDateStr):
    if df['hr'].sum() == 0:
        return None
    
    zones = [
        (0.5 * vMaxHR, 0.6 * vMaxHR, 'Z1'),
        (0.6 * vMaxHR, 0.7 * vMaxHR, 'Z2'),
        (0.7 * vMaxHR, 0.8 * vMaxHR, 'Z3'),
        (0.8 * vMaxHR, 0.9 * vMaxHR, 'Z4'),
        (0.9 * vMaxHR, 2.0 * vMaxHR, 'Z5')
    ]
    zone_times = []
    zone_labels = []
    weighted_score = 0
    
    for i, (low, high, label) in enumerate(zones):
        mask = (df['hr'] >= low) & (df['hr'] < high)
        minutes = df.loc[mask, 'time_diff'].sum() / 60
        zone_times.append(minutes)
        zone_labels.append(label)
        weighted_score += (i+1) * minutes

    plt.figure(figsize=(8, 4))
    plt.bar(zone_labels, zone_times, color=['grey', 'blue', 'green', 'orange', 'red'])
    plt.title(f"Heart Rate Zones (Effort Score: {int(weighted_score)})")
    plt.ylabel("Minutes")
    
    vImgName = f"hr_{vDateStr}.png"
    plt.savefig(os.path.join(vImagesPath, vImgName))
    plt.close()
    print(f"  > HR graph saved: {vImgName}")
    return int(weighted_score), vImgName

# 7. OPTION 1: Ghost Runner Comparison
# ---------------------------------------------------------
def create_ghost_comparison(dfMain, dfGhost, vDateStr):
    print("  > Generating Ghost Runner Battle...")
    
    # Create a clean distance grid (e.g. every 50m)
    max_dist = min(dfMain['total_dist_m'].max(), dfGhost['total_dist_m'].max())
    grid = np.arange(0, max_dist, 50)
    
    # Interpolate time for both runs onto this grid
    # FIX: Use .astype(np.int64) instead of .view(np.int64) for updated Pandas compatibility
    time_main = np.interp(grid, dfMain['total_dist_m'], dfMain['time'].astype(np.int64) // 10**9)
    time_ghost = np.interp(grid, dfGhost['total_dist_m'], dfGhost['time'].astype(np.int64) // 10**9)
    
    # Normalize start times to 0
    time_main = time_main - time_main[0]
    time_ghost = time_ghost - time_ghost[0]
    
    # Calculate Gap (Negative = Ahead of Ghost, Positive = Behind)
    gap = time_main - time_ghost
    
    plt.figure(figsize=(10, 6))
    
    # Fill logic
    plt.plot(grid, gap, color='black', linewidth=1)
    plt.fill_between(grid, gap, 0, where=(gap < 0), color='green', alpha=0.3, label='Ahead of Ghost')
    plt.fill_between(grid, gap, 0, where=(gap > 0), color='red', alpha=0.3, label='Behind Ghost')
    
    plt.axhline(0, color='black', linestyle='--')
    plt.title("Ghost Battle: Time Gap vs. Comparison Run")
    plt.ylabel("Time Gap (Seconds)")
    plt.xlabel("Distance (m)")
    plt.legend()
    
    vImgName = f"battle_{vDateStr}.png"
    plt.savefig(os.path.join(vImagesPath, vImgName))
    plt.close()
    print(f"  > Battle graph saved: {vImgName}")
    return vImgName

# 8. Create Blog Post
# ---------------------------------------------------------
def create_blog_post(vDateStr, hr_data, battle_img=None):
    vRelMap = f"../assets/maps/map_{vDateStr}.html"
    vRelGraph = f"../assets/images/analysis_{vDateStr}.png"
    vRelTable = f"../assets/tables/table_{vDateStr}.md"
    
    hr_section = ""
    if hr_data:
        score, img_name = hr_data
        hr_section = f"## Physiology\n**Effort Score:** {score}\n\n![](/assets/images/{img_name})"

    battle_section = ""
    if battle_img:
        battle_section = f"## The Ghost Battle\nComparing today's run against the ghost file.\n\n![](/assets/images/{battle_img})"

    vPostFilename = f"{vDateStr}-run.qmd"
    vFullPath = os.path.join(vPostsPath, vPostFilename)
    
    vTemplate = f"""---
title: "Run Analysis: {vDateStr}"
date: "{vDateStr}"
categories: [running, analysis]
image: {vRelGraph}
format:
  html:
    toc: true
---

## Route Map
<iframe src="{vRelMap}" width="100%" height="500px" style="border:none;"></iframe>

{battle_section}

{hr_section}

## Mechanics & Discipline
![]({vRelGraph})

## The Data Table
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
    
    print("[4/6] Generating Mechanics Analysis...")
    create_analysis_graph(dfRun, vRunDate)
    
    print("[5/6] Generating HR Analysis...")
    hr_result = create_hr_analysis(dfRun, vRunDate)
    
    battle_result = None
    if vGhostFilename:
        print("[OPTIONAL] Ghost Runner Detected...")
        dfGhost = parse_run_file(vGhostFilename)
        battle_result = create_ghost_comparison(dfRun, dfGhost, vRunDate)

    print("[6/6] Writing Blog Post...")
    create_blog_post(vRunDate, hr_result, battle_result)
    print("--- Done ---")