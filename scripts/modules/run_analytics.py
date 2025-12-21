import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import gpxpy
import pandas as pd
import folium
import matplotlib.colors as mcolors
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import os
import numpy as np
import io
import math
from tcxreader.tcxreader import TCXReader

# 0. PHYSICS & PHYSIOLOGY ENGINE
# ---------------------------------------------------------
class PhysicsEngine:
    @staticmethod
    def time_str_to_sec(t_str):
        try:
            parts = list(map(int, t_str.split(':')))
            if len(parts) == 2: return parts[0]*60 + parts[1]
            return 0
        except: return 0

    @staticmethod
    def sec_to_time_str(seconds):
        mins = int(seconds // 60)
        secs = int(seconds % 60)
        return f"{mins}:{secs:02d}"

    @staticmethod
    def adjust_pace(base_sec, multiplier):
        adj_sec = base_sec * multiplier
        return PhysicsEngine.sec_to_time_str(adj_sec)
    
    @staticmethod
    def calculate_penalty_multiplier(temp_c, wind_kmh):
        mult = 1.0
        if temp_c > 15: mult += ((temp_c - 15) * 0.003)
        if wind_kmh > 5: mult += ((wind_kmh - 5) * 0.001)
        return mult

    @staticmethod
    def generate_conditions_matrix(base_paces):
        scenarios = {
            "Neutral (15°C)": 1.0,
            "Warm (20°C)": 1.015,
            "Hot (25°C)": 1.03,
            "Rolling Hills": 1.04,
            "Headwind": 1.025
        }
        data = []
        for session, pace_range in base_paces.items():
            try:
                p_fast_str, p_slow_str = pace_range.split(' - ')
                s_fast = PhysicsEngine.time_str_to_sec(p_fast_str)
                s_slow = PhysicsEngine.time_str_to_sec(p_slow_str)
                row = {"Session Type": session}
                for cond_name, mult in scenarios.items():
                    adj_fast = PhysicsEngine.adjust_pace(s_fast, mult)
                    adj_slow = PhysicsEngine.adjust_pace(s_slow, mult)
                    row[cond_name] = f"{adj_fast} - {adj_slow}"
                data.append(row)
            except: continue
        return pd.DataFrame(data)

def calculate_gap_factor(gradient_pct):
    i = gradient_pct / 100.0
    cost = 155.4*(i**5) - 30.4*(i**4) - 43.3*(i**3) + 46.3*(i**2) + 19.5*i + 3.6
    flat_cost = 3.6
    return cost / flat_cost

def calculate_zones(df, max_hr):
    if 'hr' not in df.columns or df['hr'].isnull().all(): return None
    zones = [0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    labels = ['Z1', 'Z2', 'Z3', 'Z4', 'Z5']
    counts = []
    for i in range(5):
        low = max_hr * zones[i]
        high = max_hr * zones[i+1]
        mask = (df['hr'] >= low) & (df['hr'] < high)
        if i == 4: mask = (df['hr'] >= low) 
        counts.append(len(df[mask]))
    mins = [c / 60 for c in counts]
    return pd.DataFrame({'Zone': labels, 'Minutes': mins})

def calculate_decoupling(df):
    if 'hr' not in df.columns or 'speed_smooth' not in df.columns: return None
    # If speed is 0 (strength), return None
    if df['speed_smooth'].sum() == 0: return None
    
    mask = (df['speed_smooth'] > 1.5) & (df['hr'] > 60)
    clean_df = df[mask].copy()
    if len(clean_df) < 600: return None 
    midpoint = len(clean_df) // 2
    h1 = clean_df.iloc[:midpoint]
    h2 = clean_df.iloc[midpoint:]
    ef1 = (h1['speed_smooth'] / h1['hr']).mean()
    ef2 = (h2['speed_smooth'] / h2['hr']).mean()
    if ef1 == 0: return 0
    decoupling = ((ef1 - ef2) / ef1) * 100
    return round(decoupling, 2)

def calculate_training_paces(race_dist_km, race_time_str):
    try:
        parts = list(map(int, race_time_str.split(':')))
        if len(parts) == 2: seconds = parts[0]*60 + parts[1]
        elif len(parts) == 3: seconds = parts[0]*3600 + parts[1]*60 + parts[2]
        else: return None
    except: return None
    t1 = seconds
    d1 = race_dist_km
    def get_pace_range(dist, variance_sec=3):
        t2 = t1 * (dist / d1)**1.06
        base_pace_sec = t2 / dist
        fast_sec = base_pace_sec - variance_sec
        slow_sec = base_pace_sec + variance_sec
        def fmt(s): return f"{int(s // 60)}:{int(s % 60):02d}"
        return f"{fmt(fast_sec)} - {fmt(slow_sec)}"

    base_paces = {
        "Short Intervals (15k Pace)": get_pace_range(21.0975, variance_sec=4),
        "Medium Intervals (HM Pace)": get_pace_range(30.0, variance_sec=4),
        "Long Intervals (30k Pace)": get_pace_range(42.195, variance_sec=4)
    }
    matrix_df = PhysicsEngine.generate_conditions_matrix(base_paces)
    return base_paces, matrix_df

# 1. PARSING (FIXED FOR STRENGTH FILES)
# ---------------------------------------------------------
def parse_file(file_path, smoothing_span=15):
    if not os.path.exists(file_path): return None
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
        if df.empty: return None

        # --- SAFE GPS HANDLING ---
        # Strength files often have no Lat/Lon. We must handle this to prevent crashes.
        has_gps = False
        if 'lat' in df.columns and 'lon' in df.columns:
            if not df['lat'].isnull().all():
                has_gps = True

        df['prev_time'] = df['time'].shift(1)
        df['time_diff'] = (df['time'] - df['prev_time']).dt.total_seconds()
        
        # Filter jumps
        df['adjusted_diff'] = np.where(df['time_diff'] > 10, 0, df['time_diff'])
        df['timer_sec'] = df['adjusted_diff'].cumsum().fillna(0)
        df = df[df['time_diff'] > 0]

        if has_gps:
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
            
            df['speed_mps'] = df['dist_m'] / df['time_diff']
            df['speed_smooth'] = df['speed_mps'].ewm(span=smoothing_span, adjust=False).mean()
            
            # Elevation & GAP
            df['ele_smooth'] = df['ele'].rolling(window=smoothing_span, min_periods=1, center=True).mean()
            df['ele_change'] = df['ele_smooth'].diff()
            df['gradient_pct'] = (df['ele_change'] / df['dist_m']) * 100
            df['gradient_pct'] = df['gradient_pct'].fillna(0).clip(-25, 25)
            df['gradient_smooth'] = df['gradient_pct'].ewm(span=smoothing_span*2, adjust=False).mean()
            df['gap_factor'] = df['gradient_smooth'].apply(calculate_gap_factor)
            df['gap_speed_mps'] = df['speed_smooth'] * df['gap_factor']
        else:
            # NO GPS MODE (Strength/Treadmill) - Fill with Zeros
            df['dist_m'] = 0
            df['total_dist_m'] = 0
            df['speed_mps'] = 0
            df['speed_smooth'] = 0
            df['gap_speed_mps'] = 0
            df['ele_change'] = 0
            df['gradient_pct'] = 0
        
        df['cadence'] = df['cadence'].fillna(0)
        return df
    except Exception as e:
        print(f"Error parsing file: {e}")
        return None

# 2. VISUALS
# ---------------------------------------------------------
def generate_dashboard(df, output_path, title="Run Deep Dive", max_hr=190):
    if df['total_dist_m'].max() < 100: return # Skip dash for non-runs
    
    df['segment_100m'] = (df['total_dist_m'] // 100).astype(int)
    agg_dict = {'gap_speed_mps': 'mean', 'speed_smooth': 'mean', 'hr': 'mean', 'cadence': 'mean', 'ele_smooth': 'mean', 'total_dist_m': 'max'}
    available_cols = [c for c in agg_dict.keys() if c in df.columns]
    safe_agg = {k: agg_dict[k] for k in available_cols}
    df_agg = df.groupby('segment_100m').agg(safe_agg).reset_index()
    
    decoupling = calculate_decoupling(df)
    decoupling_str = f"{decoupling}%" if decoupling is not None else "N/A"
    zones_df = calculate_zones(df, max_hr)
    
    fig = make_subplots(
        rows=4, cols=1, shared_xaxes=False, 
        vertical_spacing=0.08, row_heights=[0.35, 0.25, 0.15, 0.25],
        specs=[[{"secondary_y": True}], [{"secondary_y": True}], [{"secondary_y": False}], [{"secondary_y": False}]]
    )
    x_axis = df_agg['total_dist_m'] / 1000 
    
    fig.add_trace(go.Scatter(x=x_axis, y=df_agg['gap_speed_mps'], name='GAP (Effort)', line=dict(color='#1f77b4', width=2), hovertemplate='%{y:.2f} m/s'), row=1, col=1, secondary_y=False)
    if 'hr' in df_agg.columns:
        fig.add_trace(go.Scatter(x=x_axis, y=df_agg['hr'], name='Heart Rate', line=dict(color='#d62728', width=1), fill='tozeroy', opacity=0.1), row=1, col=1, secondary_y=True)

    if 'cadence' in df_agg.columns:
        fig.add_trace(go.Scatter(x=x_axis, y=df_agg['cadence'], name='Cadence', line=dict(color='#9467bd', width=1.5)), row=2, col=1, secondary_y=False)

    fig.add_trace(go.Scatter(x=x_axis, y=df_agg['ele_smooth'], name='Elevation', fill='tozeroy', line=dict(color='#7f7f7f', width=1)), row=3, col=1)

    if zones_df is not None:
        colors = ['#a6cee3', '#1f78b4', '#b2df8a', '#33a02c', '#e31a1c']
        fig.add_trace(go.Bar(x=zones_df['Zone'], y=zones_df['Minutes'], marker_color=colors, name='Time in Zone'), row=4, col=1)

    fig.update_layout(height=1000, title_text=f"{title} | Decoupling: {decoupling_str}", hovermode="x unified", template="plotly_white", showlegend=False)
    fig.write_html(output_path)

def analyze_recovery(df, target_max_hr):
    max_hr_hit = df['hr'].max()
    avg_hr = int(df['hr'].mean()) 
    violation_sec = df[df['hr'] > target_max_hr]['time_diff'].sum()
    violation_mins = violation_sec / 60
    score = max(0, int(100 - violation_mins))
    status = "SUCCESS" if score == 100 else "WARNING" if score > 90 else "FAILED"
    df['minute_bin'] = (df['timer_sec'] // 60).astype(int)
    minute_stats = df.groupby('minute_bin')[['hr', 'gap_speed_mps']].mean().reset_index()
    minute_stats.columns = ['Minute', 'Avg HR', 'Avg GAP']
    return {
        "status": status, "score": score, "max_hit": max_hr_hit,
        "avg_hr": avg_hr, "violation_time_min": round(violation_mins, 1),
        "minute_data": minute_stats
    }

def create_recovery_figure(minute_df, target_hr):
    colors = ['red' if x > target_hr else 'green' for x in minute_df['Avg HR']]
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.add_trace(go.Bar(x=minute_df['Minute'], y=minute_df['Avg HR'], marker_color=colors, name='Avg HR'), secondary_y=False)
    fig.add_trace(go.Scatter(x=minute_df['Minute'], y=minute_df['Avg GAP'], line=dict(color='grey', dash='dot'), name='GAP (m/s)'), secondary_y=True)
    fig.add_shape(type="line", x0=minute_df['Minute'].min()-0.5, x1=minute_df['Minute'].max()+0.5, y0=target_hr, y1=target_hr, line=dict(color="black", width=2, dash="dash"), name="Limit")
    fig.update_layout(title=f"Heart Rate vs Effort (GAP)", xaxis_title="Minute", yaxis_title="Avg BPM")
    return fig

def create_ghost_figure(dfMain, dfGhost):
    max_dist = min(dfMain['total_dist_m'].max(), dfGhost['total_dist_m'].max())
    if max_dist == 0: return None
    grid = np.arange(0, max_dist, 10)
    time_main = np.interp(grid, dfMain['total_dist_m'], dfMain['timer_sec'])
    time_ghost = np.interp(grid, dfGhost['total_dist_m'], dfGhost['timer_sec'])
    gap = time_main - time_ghost
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=grid, y=gap, mode='lines', name='Gap (sec)', line=dict(color='black', width=2), fill='tozeroy', fillcolor='rgba(100, 100, 100, 0.2)'))
    fig.update_layout(title="Ghost Battle: Gap to Comparison Run", xaxis_title="Distance (m)", yaxis_title="Seconds (Neg=Ahead, Pos=Behind)", hovermode="x unified")
    fig.add_shape(type="line", x0=0, x1=max_dist, y0=0, y1=0, line=dict(color="red", width=1, dash="dash"))
    return fig

def analyze_intervals(df, warm_min, work_min, rest_min, reps, cool_min, buffer_sec=0, target_pace_str=None, target_hr=None, use_gap=True, ignore_hr=False, lenient_first_rep=False, temp_c=15, wind_kmh=0, surface_penalty_sec=0):
    intervals = []
    target_mps = None
    target_sec_km = 0
    
    penalty_mult = PhysicsEngine.calculate_penalty_multiplier(temp_c, wind_kmh)
    
    if target_pace_str:
        try:
            mins, secs = map(int, target_pace_str.split(':'))
            target_sec_km = (mins * 60) + secs
            target_mps = 1000 / target_sec_km
        except: pass

    total_pace_points = 0
    total_hr_points = 0
    max_pace_pts = 100 if ignore_hr else 50
    pace_points_per_rep = max_pace_pts / reps if reps > 0 else 0
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
            scoring_mps = df_int['gap_speed_mps'].mean() if use_gap else df_int['speed_smooth'].mean()
            
            gap_pace_str = "0:00"
            if df_int['gap_speed_mps'].mean() > 0:
                s = 1000 / df_int['gap_speed_mps'].mean()
                gap_pace_str = f"{int(s//60)}:{int(s%60):02d}"

            raw_pace_str = "0:00"
            if df_int['speed_smooth'].mean() > 0:
                s = 1000 / df_int['speed_smooth'].mean()
                raw_pace_str = f"{int(s//60)}:{int(s%60):02d}"
                
            ngp_str = "N/A"
            if scoring_mps > 0:
                actual_sec_km = 1000 / scoring_mps
                env_adjusted_sec = actual_sec_km / penalty_mult
                final_ngp_sec = env_adjusted_sec - surface_penalty_sec
                if final_ngp_sec < 0: final_ngp_sec = actual_sec_km
                ngp_str = f"{int(final_ngp_sec//60)}:{int(final_ngp_sec%60):02d}"
            
            rep_sec_km = 0
            if scoring_mps > 0: rep_sec_km = 1000 / scoring_mps
            
            ele_gain = df_int[df_int['ele_change'] > 0]['ele_change'].sum()
            ele_loss = df_int[df_int['ele_change'] < 0]['ele_change'].sum()
            
            pace_points_earned = 0
            rep_pace_score = "N/A"
            if target_mps and rep_sec_km > 0:
                diff = rep_sec_km - target_sec_km 
                abs_diff = abs(diff)
                
                threshold_perfect = 5
                if i == 1 and lenient_first_rep and diff > 0: threshold_perfect = 15

                if abs_diff <= threshold_perfect: pace_points_earned = pace_points_per_rep * 1.0; rep_pace_score = "Target"
                elif abs_diff <= (threshold_perfect + 5): pace_points_earned = pace_points_per_rep * 0.66; rep_pace_score = "Close"
                elif abs_diff <= (threshold_perfect + 10): pace_points_earned = pace_points_per_rep * 0.33; rep_pace_score = "Okay"
                else: rep_pace_score = "Miss"
                total_pace_points += pace_points_earned

            hr_score_pct = 0
            avg_hr_rep = int(df_int['hr'].mean()) if not df['hr'].isna().all() else 0
            if target_hr and not ignore_hr:
                under_hr = df_int[df_int['hr'] <= target_hr]
                hr_score_pct = (len(under_hr) / len(df_int)) * 100
                total_hr_points += hr_score_pct 

            intervals.append({
                "Rep": i, "Avg HR": avg_hr_rep if not ignore_hr else "N/A", 
                "GAP Pace": gap_pace_str, 
                "Raw Pace": raw_pace_str,
                "NGP (Effort)": ngp_str, 
                "Elev Net": f"+{int(ele_gain)}/{int(ele_loss)}m",
                "HR Compliance %": f"{int(hr_score_pct)}%" if not ignore_hr else "N/A", 
                "Pace Rating": rep_pace_score
            })
        t_cursor = rep_end_abs + (rest_min * 60)
    
    if ignore_hr: final_hr_score = 0; avg_hr_compliance = 0
    else: avg_hr_compliance = total_hr_points / reps if reps > 0 else 0; final_hr_score = (avg_hr_compliance / 100) * 50
    final_total_score = int(total_pace_points + final_hr_score)
    scores = {"Total": final_total_score, "Pace Pts": int(total_pace_points), "HR Pts": int(final_hr_score), "Avg HR Compliance": int(avg_hr_compliance)}
    return pd.DataFrame(intervals), target_mps, scores

def create_interval_figure(df, intervals_df, target_pace_str, target_mps, warm_min, work_min, rest_min, reps, use_gap=True):
    if not target_mps: return None
    mins, secs = map(int, target_pace_str.split(':'))
    base_sec_km = (mins * 60) + secs
    start_trim = warm_min * 60
    end_trim = start_trim + (reps * (work_min + rest_min) * 60)
    df_trim = df[(df['timer_sec'] >= start_trim) & (df['timer_sec'] <= end_trim)].copy()
    speed_target = target_mps
    s_plus_10 = 1000 / (base_sec_km - 10); s_minus_10 = 1000 / (base_sec_km + 10)
    s_plus_15 = 1000 / (base_sec_km - 15); s_minus_15 = 1000 / (base_sec_km + 15)

    fig = go.Figure()
    x0, x1 = start_trim, end_trim
    fig.add_shape(type="rect", x0=x0, x1=x1, y0=s_minus_15, y1=s_plus_15, fillcolor="rgba(255, 165, 0, 0.15)", line_width=0, layer="below", name="Orange")
    fig.add_shape(type="rect", x0=x0, x1=x1, y0=s_minus_10, y1=s_plus_10, fillcolor="rgba(0, 255, 0, 0.25)", line_width=0, layer="below", name="Green")
    fig.add_shape(type="line", x0=x0, x1=x1, y0=speed_target, y1=speed_target, line=dict(color="green", width=2, dash="dash"), name="Target")
    
    if use_gap:
        fig.add_trace(go.Scatter(x=df_trim['timer_sec'], y=df_trim['gap_speed_mps'], mode='lines', name='GAP', line=dict(color='black', width=1.5, shape='spline')))
    else:
        fig.add_trace(go.Scatter(x=df_trim['timer_sec'], y=df_trim['speed_smooth'], mode='lines', name='Raw', line=dict(color='black', width=1.5, shape='spline')))

    for i, row in intervals_df.iterrows():
        rep_num = int(row['Rep'])
        w_start = (warm_min * 60) + ((rep_num - 1) * ((work_min + rest_min) * 60))
        w_end = w_start + (work_min * 60)
        fig.add_vrect(x0=w_start, x1=w_end, fillcolor="grey", opacity=0.05, layer="below")
        p_str = row.get('GAP Pace') if use_gap else row.get('Raw Pace')
        if not p_str: p_str = row.get('Pace', '0:00')
        p_min, p_sec = map(int, p_str.split(':'))
        if ((p_min*60)+p_sec) > 0:
            rep_speed = 1000 / ((p_min*60)+p_sec)
            fig.add_trace(go.Scatter(x=[w_start, w_end], y=[rep_speed, rep_speed], mode='lines', line=dict(color='blue', width=3), showlegend=False))

    y_center = speed_target
    y_range = (s_plus_15 - s_minus_15) * 1.5
    fig.update_yaxes(range=[y_center - y_range, y_center + y_range], title="Speed (m/s)", showgrid=False)
    fig.update_layout(title=f"Interval Discipline", hovermode="x unified")
    return fig

def create_infographic(run_type, stats, score, score_label, df_trace=None, trace_col=None, target_line=None, intervals_df=None, warm_min=0, work_min=0, rest_min=0, reps=0, use_gap=True):
    fig = plt.figure(figsize=(6, 5), facecolor='#f8f9fa')
    gs = fig.add_gridspec(2, 1, height_ratios=[1, 0.6])
    
    ax_text = fig.add_subplot(gs[0])
    ax_text.axis('off')
    plt.text(0.05, 0.85, run_type, fontsize=16, fontweight='bold', color='#333', transform=ax_text.transAxes)
    plt.text(0.05, 0.75, "Run Analysis Tool", fontsize=10, color='grey', transform=ax_text.transAxes)
    
    circle_color = '#28a745' if score >= 80 else '#ffc107' if score >= 50 else '#dc3545'
    circle = plt.Circle((0.85, 0.7), 0.18, color=circle_color, transform=ax_text.transAxes, zorder=10)
    ax_text.add_patch(circle)
    plt.text(0.85, 0.67, str(score), fontsize=22, fontweight='bold', color='white', ha='center', va='center', transform=ax_text.transAxes, zorder=11)
    plt.text(0.85, 0.45, score_label, fontsize=7, color=circle_color, ha='center', transform=ax_text.transAxes)
    
    y_pos = 0.50
    for k, v in stats.items():
        plt.text(0.05, y_pos, k.upper(), fontsize=8, color='grey', transform=ax_text.transAxes)
        plt.text(0.35, y_pos, v, fontsize=10, fontweight='bold', transform=ax_text.transAxes)
        y_pos -= 0.15

    if df_trace is not None and trace_col in df_trace.columns:
        ax_chart = fig.add_subplot(gs[1])
        ax_chart.set_facecolor='#f8f9fa'
        for spine in ax_chart.spines.values(): spine.set_visible(False)
        ax_chart.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)
        
        if intervals_df is not None and target_line:
            start_trim = warm_min * 60
            end_trim = start_trim + (reps * (work_min + rest_min) * 60)
            df_plot = df_trace[(df_trace.index >= start_trim) & (df_trace.index <= end_trim)]
            
            base_speed = target_line
            sec_km = 1000 / base_speed
            s_minus_15 = 1000 / (sec_km + 15); s_plus_15 = 1000 / (sec_km - 15)
            s_minus_10 = 1000 / (sec_km + 10); s_plus_10 = 1000 / (sec_km - 10)
            
            ax_chart.axhspan(s_minus_15, s_plus_15, color='orange', alpha=0.15)
            ax_chart.axhspan(s_minus_10, s_plus_10, color='green', alpha=0.25)
            ax_chart.axhline(base_speed, color='green', linestyle='--', linewidth=1)
            ax_chart.plot(df_plot.index, df_plot[trace_col], color='black', linewidth=1)
            
            for _, row in intervals_df.iterrows():
                rep_num = int(row['Rep'])
                w_start = (warm_min * 60) + ((rep_num - 1) * ((work_min + rest_min) * 60))
                w_end = w_start + (work_min * 60)
                p_str = row.get('GAP Pace') if use_gap else row.get('Raw Pace')
                if not p_str: p_str = row.get('Pace', '0:00')
                p_min, p_sec = map(int, p_str.split(':'))
                rep_speed = 1000 / ((p_min*60) + p_sec)
                ax_chart.plot([w_start, w_end], [rep_speed, rep_speed], color='blue', linewidth=2)
            
            y_range = (s_plus_15 - s_minus_15) * 1.5
            ax_chart.set_ylim(base_speed - y_range, base_speed + y_range)

        else:
            ax_chart.plot(df_trace.index, df_trace[trace_col], color='#444', linewidth=1.5)
            ax_chart.fill_between(df_trace.index, df_trace[trace_col], alpha=0.1, color='#444')
            if target_line: ax_chart.axhline(y=target_line, color=circle_color, linestyle='--', linewidth=1, alpha=0.8)

    buf = io.BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight', dpi=150)
    buf.seek(0)
    plt.close(fig)
    return buf

def create_pace_card(race_dist, race_time, base_paces):
    fig, ax = plt.subplots(figsize=(6, 4), facecolor='white')
    ax.axis('off')
    
    plt.text(0.5, 0.9, "TRAINING TARGETS", ha='center', fontsize=16, fontweight='bold', color='#333')
    plt.text(0.5, 0.8, f"Based on: {race_dist} @ {race_time}", ha='center', fontsize=10, color='gray')
    
    y_pos = 0.6
    plt.text(0.1, 0.65, "SESSION", fontsize=9, fontweight='bold', color='#555')
    plt.text(0.6, 0.65, "PACE RANGE", fontsize=9, fontweight='bold', color='#555')
    plt.axhline(y=0.63, xmin=0.05, xmax=0.95, color='#ddd', linewidth=1)

    for k, v in base_paces.items():
        name = k.split(" (")[0]
        plt.text(0.1, y_pos, name, fontsize=11, fontweight='bold')
        plt.text(0.6, y_pos, v, fontsize=11, color='#1f77b4', fontweight='bold')
        y_pos -= 0.15
    
    buf = io.BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight', dpi=150)
    buf.seek(0)
    plt.close(fig)
    return buf

def create_recovery_card(stats, score, score_label):
    fig = plt.figure(figsize=(6, 4), facecolor='white')
    ax = fig.add_subplot(111)
    ax.axis('off')
    
    plt.text(0.05, 0.9, "Recovery Session", fontsize=14, fontweight='bold', color='#333')
    plt.text(0.05, 0.82, "Compliance Report", fontsize=9, color='gray')

    color = '#28a745' if score == 100 else '#dc3545'
    plt.text(0.85, 0.85, str(score), fontsize=24, fontweight='bold', color=color, ha='center')
    plt.text(0.85, 0.75, score_label, fontsize=7, color='gray', ha='center')

    plt.text(0.1, 0.6, f"Max HR Hit: {stats['max_hit']} bpm", fontsize=10)
    plt.text(0.1, 0.5, f"Avg HR: {stats['avg_hr']} bpm", fontsize=10)
    plt.text(0.1, 0.4, f"Violation Time: {stats['violation_time_min']} min", fontsize=10, color=color if stats['violation_time_min']==0 else 'red')
    
    plt.axhline(y=0.3, xmin=0.05, xmax=0.95, color='#ddd', linewidth=1)
    plt.text(0.5, 0.2, "Goal: Stay below Cap (Zone 1/2)", ha='center', fontsize=8, style='italic')

    buf = io.BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight', dpi=150)
    buf.seek(0)
    plt.close(fig)
    return buf