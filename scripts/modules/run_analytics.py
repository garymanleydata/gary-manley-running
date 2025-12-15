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

# 1. PARSING
# ---------------------------------------------------------
def parse_file(file_path, smoothing_span=15):
    print(f"  [Engine] Parsing {os.path.basename(file_path)}...")
    if not os.path.exists(file_path):
        print("  ! Error: File does not exist.")
        return None
        
    ext = os.path.splitext(file_path)[1].lower()
    data = []

    try:
        if ext == '.gpx':
            with open(file_path, 'r') as gpx_file:
                gpx = gpxpy.parse(gpx_file)
            for track in gpx.tracks:
                for segment in track.segments:
                    for point in segment.points:
                        hr, cad = None, None
                        if point.extensions:
                            for ext_node in point.extensions:
                                if 'TrackPointExtension' in ext_node.tag:
                                    for child in ext_node:
                                        if 'hr' in child.tag: hr = int(child.text)
                                        if 'cad' in child.tag: cad = int(child.text)
                        data.append({'time': point.time, 'lat': point.latitude, 'lon': point.longitude, 'ele': point.elevation, 'hr': hr, 'cadence': cad})

        elif ext == '.tcx':
            tcx_reader = TCXReader()
            activity = tcx_reader.read(file_path)
            for tp in activity.trackpoints:
                data.append({'time': tp.time, 'lat': tp.latitude, 'lon': tp.longitude, 'ele': tp.elevation, 'hr': tp.hr_value, 'cadence': tp.cadence})

        df = pd.DataFrame(data)
        
        df['prev_time'] = df['time'].shift(1)
        df['prev_lat'] = df['lat'].shift(1)
        
        R = 6371000 
        phi1 = np.radians(df['lat'])
        phi2 = np.radians(df['prev_lat'])
        dphi = np.radians(df['prev_lat'] - df['lat'])
        dlambda = np.radians(df['lon'].shift(1) - df['lon'])
        a = np.sin(dphi/2)**2 + np.cos(phi1)*np.cos(phi2) * np.sin(dlambda/2)**2
        c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1-a))
        df['dist_m'] = R * c
        df['dist_m'] = df['dist_m'].fillna(0)
        df['total_dist_m'] = df['dist_m'].cumsum()
        
        df['time_diff'] = (df['time'] - df['prev_time']).dt.total_seconds()
        
        # Smart Timer
        df['adjusted_diff'] = np.where(df['time_diff'] > 10, 0, df['time_diff'])
        df['timer_sec'] = df['adjusted_diff'].cumsum().fillna(0)
        
        df = df[df['time_diff'] > 0] 
        df['speed_mps'] = df['dist_m'] / df['time_diff']
        
        # Smoothing
        df['speed_smooth'] = df['speed_mps'].ewm(span=smoothing_span, adjust=False).mean()
        
        # Cadence Fix
        df['cadence'] = df['cadence'].fillna(0)
        if df['cadence'].max() > 0 and df['cadence'].max() < 130:
            df['cadence'] = df['cadence'] * 2
            
        df['stride_len'] = np.where((df['cadence'] > 60) & (df['speed_smooth'] > 0.5), 
                                    df['speed_smooth'] / (df['cadence'] / 60), np.nan)
        
        return df
    except Exception as e:
        print(f"  ! Error parsing file: {e}")
        return None

# 2. STANDARD VISUALS
# ---------------------------------------------------------
def generate_map(df, output_path):
    m = folium.Map(location=[df.iloc[0]['lat'], df.iloc[0]['lon']], zoom_start=14, tiles='CartoDB positron')
    avg_speed = df['speed_smooth'].mean()
    cmap = plt.get_cmap('RdYlGn')
    df_sample = df.iloc[::2]
    
    for i in range(1, len(df_sample)):
        norm = max(0, min(1, (df_sample.iloc[i]['speed_smooth'] - (avg_speed*0.5))/avg_speed))
        color = mcolors.to_hex(cmap(norm))
        folium.PolyLine(
            locations=[(df_sample.iloc[i-1]['lat'], df_sample.iloc[i-1]['lon']), 
                       (df_sample.iloc[i]['lat'], df_sample.iloc[i]['lon'])],
            color=color, weight=5, opacity=0.8
        ).add_to(m)
    m.save(output_path)

def generate_dashboard(df, output_path, title="Run Analysis"):
    df['segment_100m'] = (df['total_dist_m'] // 100).astype(int)
    agg_dict = {'time_diff': 'sum', 'stride_len': 'mean', 'cadence': 'mean'}
    df_agg = df.groupby('segment_100m').agg(agg_dict).reset_index()
    
    avg_time = df_agg['time_diff'].mean()
    df_agg['variance'] = df_agg['time_diff'] - avg_time
    colors = ['#2ca02c' if v <= 0 else '#d62728' for v in df_agg['variance']]

    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, specs=[[{"secondary_y": False}], [{"secondary_y": True}]])
    fig.add_trace(go.Bar(x=df_agg['segment_100m'], y=df_agg['variance'], marker_color=colors, name='Pace Var'), row=1, col=1)
    fig.add_trace(go.Scatter(x=df_agg['segment_100m'], y=df_agg['stride_len'], name='Stride (m)', line=dict(color='blue')), row=2, col=1)
    fig.add_trace(go.Scatter(x=df_agg['segment_100m'], y=df_agg['cadence'], name='Cadence', line=dict(color='orange', dash='dot')), row=2, col=1, secondary_y=True)
    fig.update_layout(height=600, title_text=title, hovermode="x unified")
    fig.write_html(output_path)

# 3. ANALYSIS: RECOVERY (Updated Scoring)
# ---------------------------------------------------------
def analyze_recovery(df, target_max_hr):
    max_hr_hit = df['hr'].max()
    
    # Calculate seconds spent ABOVE target
    violation_sec = df[df['hr'] > target_max_hr]['time_diff'].sum()
    violation_mins = violation_sec / 60
    
    # Scoring: 100 - (Minutes Over)
    # e.g. 5 mins over = 95/100
    score = max(0, int(100 - violation_mins))
    
    status = "SUCCESS" if score == 100 else "WARNING" if score > 90 else "FAILED"
    
    return {
        "status": status,
        "score": score,
        "max_hit": max_hr_hit,
        "violation_time_min": round(violation_mins, 1)
    }

# 4. GHOST BATTLE (Restored)
# ---------------------------------------------------------
def generate_ghost_plot(dfMain, dfGhost, output_path):
    print("  [Engine] Generating Ghost Comparison...")
    
    # Create Distance Grid
    max_dist = min(dfMain['total_dist_m'].max(), dfGhost['total_dist_m'].max())
    grid = np.arange(0, max_dist, 50)
    
    # Interpolate Times
    time_main = np.interp(grid, dfMain['total_dist_m'], dfMain['time'].astype(np.int64) // 10**9)
    time_ghost = np.interp(grid, dfGhost['total_dist_m'], dfGhost['time'].astype(np.int64) // 10**9)
    
    # Normalize start to 0
    time_main -= time_main[0]
    time_ghost -= time_ghost[0]
    
    # Gap: Negative = Ahead, Positive = Behind
    gap = time_main - time_ghost
    
    fig = go.Figure()
    
    # Main Line
    fig.add_trace(go.Scatter(
        x=grid, y=gap,
        mode='lines', name='Gap (sec)',
        line=dict(color='black', width=2),
        fill='tozeroy',
        # Simple gradient trick not easy in simple Plotly, using static color for now
        fillcolor='rgba(100, 100, 100, 0.2)'
    ))
    
    fig.update_layout(
        title="Ghost Battle: Gap to Comparison Run",
        xaxis_title="Distance (m)",
        yaxis_title="Seconds (Neg=Ahead, Pos=Behind)",
        hovermode="x unified"
    )
    
    # Add colored regions? Simple horizontal line
    fig.add_shape(type="line", x0=0, x1=max_dist, y0=0, y1=0, line=dict(color="red", width=1))
    
    fig.write_html(output_path)

# 5. ANALYSIS: INTERVALS
# ---------------------------------------------------------
def analyze_intervals(df, warm_min, work_min, rest_min, reps, cool_min, buffer_sec=0, target_pace_str=None, target_hr=None):
    intervals = []
    
    target_mps = None
    target_sec_km = 0
    if target_pace_str:
        try:
            mins, secs = map(int, target_pace_str.split(':'))
            target_sec_km = (mins * 60) + secs
            target_mps = 1000 / target_sec_km
        except: pass

    total_pace_points = 0
    total_hr_points = 0
    pace_points_per_rep = 50 / reps if reps > 0 else 0
    
    t_cursor = warm_min * 60 
    
    for i in range(1, reps + 1):
        rep_start_abs = t_cursor
        rep_end_abs = rep_start_abs + (work_min * 60)
        
        calc_start = rep_start_abs + buffer_sec
        calc_end = rep_end_abs - buffer_sec
        if calc_end <= calc_start: calc_start, calc_end = rep_start_abs, rep_end_abs

        mask = (df['timer_sec'] >= calc_start) & (df['timer_sec'] < calc_end)
        df_int = df.loc[mask]
        
        if not df_int.empty:
            avg_mps = df_int['speed_smooth'].mean()
            pace_str = "0:00"
            rep_sec_km = 0
            if avg_mps > 0:
                rep_sec_km = 1000 / avg_mps
                pace_str = f"{int(rep_sec_km//60)}:{int(rep_sec_km%60):02d}"
            
            pace_points_earned = 0
            rep_pace_score = "N/A"
            if target_mps and rep_sec_km > 0:
                diff = abs(rep_sec_km - target_sec_km)
                if diff <= 5:
                    pace_points_earned = pace_points_per_rep * 1.0
                    rep_pace_score = "Target (+/- 5s)"
                elif diff <= 10:
                    pace_points_earned = pace_points_per_rep * 0.66
                    rep_pace_score = "Close (+/- 10s)"
                elif diff <= 15:
                    pace_points_earned = pace_points_per_rep * 0.33
                    rep_pace_score = "Okay (+/- 15s)"
                else:
                    rep_pace_score = "Miss (> 15s)"
                total_pace_points += pace_points_earned

            hr_score_pct = 0
            if target_hr:
                under_hr = df_int[df_int['hr'] <= target_hr]
                hr_score_pct = (len(under_hr) / len(df_int)) * 100
                total_hr_points += hr_score_pct 

            intervals.append({
                "Rep": i,
                "Avg HR": int(df_int['hr'].mean()),
                "Max HR": int(df_int['hr'].max()),
                "Pace": pace_str,
                "HR Compliance %": f"{int(hr_score_pct)}%",
                "Pace Rating": rep_pace_score
            })
        
        t_cursor = rep_end_abs + (rest_min * 60)
    
    avg_hr_compliance = total_hr_points / reps if reps > 0 else 0
    final_hr_score = (avg_hr_compliance / 100) * 50
    final_total_score = int(total_pace_points + final_hr_score)
    
    scores = {
        "Total": final_total_score,
        "Pace Pts": int(total_pace_points),
        "HR Pts": int(final_hr_score),
        "Avg HR Compliance": int(avg_hr_compliance)
    }

    return pd.DataFrame(intervals), target_mps, scores

# 6. DISCIPLINE GRAPH (Intervals)
# ---------------------------------------------------------
def generate_interval_graph(df, intervals_df, output_path, target_pace_str, target_mps, warm_min, work_min, rest_min, reps):
    if not target_mps: return
    
    mins, secs = map(int, target_pace_str.split(':'))
    base_sec_km = (mins * 60) + secs
    
    speed_target = target_mps
    speed_plus_10s = 1000 / (base_sec_km - 10) 
    speed_minus_10s = 1000 / (base_sec_km + 10) 
    speed_plus_15s = 1000 / (base_sec_km - 15) 
    speed_minus_15s = 1000 / (base_sec_km + 15)

    fig = go.Figure()
    max_time = df['timer_sec'].max()
    
    fig.add_shape(type="rect", x0=0, x1=max_time, y0=speed_minus_15s, y1=speed_plus_15s,
                  fillcolor="rgba(255, 165, 0, 0.15)", line_width=0, layer="below", name="Orange Band")
    fig.add_shape(type="rect", x0=0, x1=max_time, y0=speed_minus_10s, y1=speed_plus_10s,
                  fillcolor="rgba(0, 255, 0, 0.25)", line_width=0, layer="below", name="Green Band")
    fig.add_shape(type="line", x0=0, x1=max_time, y0=speed_target, y1=speed_target,
                  line=dict(color="green", width=2, dash="dash"), name="Target Pace")

    fig.add_trace(go.Scatter(x=df['timer_sec'], y=df['speed_smooth'], 
                             mode='lines', name='Instant Pace (Smoothed)', 
                             line=dict(color='black', width=1.5, shape='spline'), 
                             hovertemplate='%{y:.2f} m/s <extra></extra>'))

    t_cursor = warm_min * 60
    for i, row in intervals_df.iterrows():
        rep_num = int(row['Rep'])
        w_start = (warm_min * 60) + ((rep_num - 1) * ((work_min + rest_min) * 60))
        w_end = w_start + (work_min * 60)
        
        fig.add_vrect(x0=w_start, x1=w_end, fillcolor="grey", opacity=0.05, layer="below", annotation_text=f"R{rep_num}")
        
        p_min, p_sec = map(int, row['Pace'].split(':'))
        rep_sec_km = (p_min * 60) + p_sec
        if rep_sec_km > 0:
            rep_avg_speed = 1000 / rep_sec_km
            fig.add_trace(go.Scatter(
                x=[w_start, w_end], y=[rep_avg_speed, rep_avg_speed],
                mode='lines', line=dict(color='blue', width=3),
                name='Rep Avg' if i == 0 else None, showlegend=(i == 0),
                hovertemplate=f"Rep {rep_num} Avg: {row['Pace']}/km<extra></extra>"
            ))

    y_center = speed_target
    y_range = (speed_plus_15s - speed_minus_15s) * 1.5
    fig.update_yaxes(range=[y_center - y_range, y_center + y_range], title="Speed (m/s)", showgrid=False)
    fig.update_layout(title=f"Interval Discipline: Target {target_pace_str}", hovermode="x unified",
                      legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1))
    
    fig.write_html(output_path)