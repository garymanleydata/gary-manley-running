import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import os
import sys

# Add scripts folder to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'scripts'))
from modules import run_analytics

# --- PAGE CONFIG ---
st.set_page_config(page_title="Run Analytics Tool", layout="wide", page_icon="🏃‍♂️")

st.title("🏃‍♂️ Advanced Run Analyzer")
st.markdown("Upload your **TCX** or **GPX** file for professional-grade analysis.")

# --- SIDEBAR CONFIG ---
st.sidebar.header("Data Source")

# 1. Main File Source
use_demo = st.sidebar.checkbox("⚡ Try with Demo Data")
uploaded_file = None

if use_demo:
    # Auto-find the latest file in your data folder to use as demo
    data_dir = os.path.join(os.path.dirname(__file__), 'data/runs')
    if os.path.exists(data_dir):
        files = [f for f in os.listdir(data_dir) if f.lower().endswith(('.tcx', '.gpx'))]
        if files:
            latest = max([os.path.join(data_dir, f) for f in files], key=os.path.getctime)
            uploaded_file = open(latest, 'rb') # Open as bytes to mimic uploader
            st.sidebar.success(f"Loaded: {os.path.basename(latest)}")
        else:
            st.sidebar.error("No demo files found in data/runs/")
else:
    uploaded_file = st.sidebar.file_uploader("Upload Run File", type=['tcx', 'gpx'])

# 2. Ghost File Source
st.sidebar.divider()
ghost_file_upload = st.sidebar.file_uploader("Optional: Ghost File (Comparison)", type=['tcx', 'gpx'])

# 3. Settings
st.sidebar.divider()
smoothing = st.sidebar.slider("GPS Smoothing (seconds)", 0, 30, 15)


# --- CACHED PARSER ---
@st.cache_data
def load_data(file_bytes, file_name, smooth_sec):
    # FIX: Ensure we only use the filename, not the full C:\ path
    safe_name = os.path.basename(file_name) 
    temp_filename = f"temp_{safe_name}"
    
    with open(temp_filename, "wb") as f:
        f.write(file_bytes)
    
    try:
        df = run_analytics.parse_file(temp_filename, smoothing_span=smooth_sec)
        return df
    finally:
        if os.path.exists(temp_filename):
            os.remove(temp_filename)

# --- MAIN LOGIC ---
if uploaded_file is not None:
    # Handle file reading depending on source (Demo=FileObject, Upload=StreamlitBuffer)
    file_bytes = uploaded_file.read() if hasattr(uploaded_file, 'read') else uploaded_file.getvalue()
    
    # FIX: Ensure we pass a name attribute even if using a raw file object
    file_name = getattr(uploaded_file, 'name', 'demo_file.tcx')

    with st.spinner("Processing Main Run..."):
        df = load_data(file_bytes, file_name, smoothing)

    # Process Ghost if exists
    df_ghost = None
    if ghost_file_upload is not None:
        with st.spinner("Processing Ghost Run..."):
            df_ghost = load_data(ghost_file_upload.getvalue(), ghost_file_upload.name, smoothing)

    if df is not None:
        # Header Stats
        run_date = df['time'].iloc[0].strftime('%Y-%m-%d %H:%M')
        dist_km = df['total_dist_m'].max() / 1000
        duration_min = df['timer_sec'].max() / 60
        
        c1, c2, c3 = st.columns(3)
        c1.metric("Date", run_date)
        c2.metric("Distance", f"{dist_km:.2f} km")
        c3.metric("Duration", f"{int(duration_min)} mins")
        
        st.divider()

        # Mode Selection
        modes = ["Standard", "Intervals", "Recovery"]
        if df_ghost is not None:
            modes.append("Ghost Battle")
            
        mode = st.radio("Select Analysis Mode:", modes, horizontal=True)

        # === MODE: GHOST BATTLE ===
        if mode == "Ghost Battle":
            st.subheader("👻 Ghost Battle Analysis")
            if df_ghost is not None:
                fig = run_analytics.create_ghost_figure(df, df_ghost)
                if fig:
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Gap Stats
                    main_time = df['timer_sec'].max()
                    ghost_time = df_ghost['timer_sec'].max()
                    diff = main_time - ghost_time
                    
                    g1, g2 = st.columns(2)
                    g1.metric("Your Time", f"{int(main_time//60)}:{int(main_time%60):02d}")
                    g2.metric("Ghost Time", f"{int(ghost_time//60)}:{int(ghost_time%60):02d}", 
                              delta=f"{-diff:.1f} sec" if diff < 0 else f"+{diff:.1f} sec",
                              delta_color="inverse") 
            else:
                st.warning("Please upload a Ghost file in the sidebar to use this mode.")

        # === MODE: INTERVALS ===
        elif mode == "Intervals":
            st.subheader("Norwegian Singles Analysis")
            with st.form("interval_config"):
                c1, c2, c3, c4 = st.columns(4)
                warm = c1.number_input("Warmup (min)", value=15)
                reps = c2.number_input("Reps", value=10, step=1)
                work = c3.number_input("Work (min)", value=3.0)
                rest = c4.number_input("Rest (min)", value=1.0)
                
                c5, c6, c7 = st.columns(3)
                cool = c5.number_input("Cool (min)", value=10)
                buffer = c6.number_input("Buffer (sec)", value=15)
                target_pace = c7.text_input("Target Pace (MM:SS)", value="3:45")
                target_hr = c5.number_input("HR Cap", value=170)
                
                analyze_btn = st.form_submit_button("Run Analysis")

            if analyze_btn:
                df_ints, target_mps, scores = run_analytics.analyze_intervals(
                    df, warm, work, rest, reps, cool, buffer, target_pace, target_hr
                )
                
                sc1, sc2, sc3 = st.columns(3)
                sc1.metric("Session Score", f"{scores['Total']}/100")
                sc2.metric("Pace Score", f"{scores['Pace Pts']}/50")
                sc3.metric("HR Score", f"{scores['HR Pts']}/50")
                
                st.dataframe(df_ints, use_container_width=True)
                
                fig = run_analytics.create_interval_figure(
                    df, df_ints, target_pace, target_mps, warm, work, rest, reps
                )
                if fig:
                    st.plotly_chart(fig, use_container_width=True)

        # === MODE: RECOVERY ===
        elif mode == "Recovery":
            st.subheader("Recovery Discipline")
            target_rec_hr = st.number_input("Target Max HR", value=145)
            # FIX: Renamed button
            if st.button("Analyze Recovery"):
                res = run_analytics.analyze_recovery(df, target_rec_hr)
                st.metric("Recovery Score", f"{res['score']}/100", delta=res['status'])
                st.write(f"Minutes Over Cap: {res['violation_time_min']}")

        # === MODE: STANDARD ===
        else:
            st.subheader("Performance Data")
            st.line_chart(df, x='total_dist_m', y=['speed_smooth', 'hr'])
            
else:
    st.info("👈 Upload a .tcx/.gpx file or click 'Try with Demo Data' to start.")