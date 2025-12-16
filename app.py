import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import os
import sys

# Add scripts folder to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'scripts'))
from modules import run_analytics

st.set_page_config(page_title="Run Analytics Tool", layout="wide", page_icon="🏃‍♂️")

st.title("🏃‍♂️ Advanced Run Analyzer")
st.markdown("Upload your **TCX** or **GPX** file for professional-grade analysis.")

# --- SIDEBAR ---
st.sidebar.header("Data Source")
use_demo = st.sidebar.checkbox("⚡ Try with Demo Data")
uploaded_file = None

if use_demo:
    data_dir = os.path.join(os.path.dirname(__file__), 'data/runs')
    if os.path.exists(data_dir):
        files = [f for f in os.listdir(data_dir) if f.lower().endswith(('.tcx', '.gpx'))]
        if files:
            latest = max([os.path.join(data_dir, f) for f in files], key=os.path.getctime)
            uploaded_file = open(latest, 'rb') 
            st.sidebar.success(f"Loaded: {os.path.basename(latest)}")
else:
    uploaded_file = st.sidebar.file_uploader("Upload Run File", type=['tcx', 'gpx'])

st.sidebar.divider()
ghost_file_upload = st.sidebar.file_uploader("Optional: Ghost File", type=['tcx', 'gpx'])
st.sidebar.divider()
smoothing = st.sidebar.slider("GPS Smoothing", 0, 30, 15)


@st.cache_data
def load_data(file_bytes, file_name, smooth_sec):
    safe_name = os.path.basename(file_name) 
    temp_filename = f"temp_{safe_name}"
    with open(temp_filename, "wb") as f:
        f.write(file_bytes)
    try:
        df = run_analytics.parse_file(temp_filename, smoothing_span=smooth_sec)
        return df
    finally:
        if os.path.exists(temp_filename): os.remove(temp_filename)

if 'int_results' not in st.session_state: st.session_state['int_results'] = None
if 'rec_results' not in st.session_state: st.session_state['rec_results'] = None

if uploaded_file is not None:
    file_bytes = uploaded_file.read() if hasattr(uploaded_file, 'read') else uploaded_file.getvalue()
    file_name = getattr(uploaded_file, 'name', 'demo_file.tcx')

    with st.spinner("Processing Main Run..."):
        df = load_data(file_bytes, file_name, smoothing)

    df_ghost = None
    if ghost_file_upload is not None:
        with st.spinner("Processing Ghost Run..."):
            df_ghost = load_data(ghost_file_upload.getvalue(), ghost_file_upload.name, smoothing)

    if df is not None:
        run_date = df['time'].iloc[0].strftime('%Y-%m-%d %H:%M')
        dist_km = df['total_dist_m'].max() / 1000
        duration_min = df['timer_sec'].max() / 60
        c1, c2, c3 = st.columns(3)
        c1.metric("Date", run_date)
        c2.metric("Distance", f"{dist_km:.2f} km")
        c3.metric("Duration", f"{int(duration_min)} mins")
        st.divider()

        modes = ["Standard", "Intervals", "Recovery"]
        if df_ghost is not None: modes.append("Ghost Battle")
        mode = st.radio("Select Analysis Mode:", modes, horizontal=True)

        if mode == "Intervals":
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
                
                if st.form_submit_button("Run Analysis"):
                    df_ints, target_mps, scores = run_analytics.analyze_intervals(
                        df, warm, work, rest, reps, cool, buffer, target_pace, target_hr
                    )
                    st.session_state['int_results'] = {
                        'df_ints': df_ints, 'target_mps': target_mps, 'scores': scores,
                        'params': (target_pace, warm, work, rest, reps, target_hr)
                    }

            if st.session_state['int_results']:
                res = st.session_state['int_results']
                scores = res['scores']
                df_ints = res['df_ints']
                target_pace, warm, work, rest, reps, target_hr = res['params']
                
                sc1, sc2, sc3 = st.columns(3)
                sc1.metric("Session Score", f"{scores['Total']}/100")
                sc2.metric("Pace Score", f"{scores['Pace Pts']}/50")
                sc3.metric("HR Score", f"{scores['HR Pts']}/50")
                
                # Set Rep as index to remove the 0,1,2 numbering
                st.dataframe(df_ints.set_index('Rep'), use_container_width=True)
                
                fig = run_analytics.create_interval_figure(df, df_ints, target_pace, res['target_mps'], warm, work, rest, reps)
                if fig: st.plotly_chart(fig, use_container_width=True)

                st.divider()
                st.subheader("📤 Export Results")
                e1, e2 = st.columns(2)
                
                csv = df_ints.to_csv(index=False).encode('utf-8')
                e1.download_button("Download Data (CSV)", csv, "intervals.csv", "text/csv")
                
                stats = {"Reps": f"{reps} x {work} min", "Target Pace": target_pace, "Target HR": f"<{target_hr} bpm"}
                
                # We reset the index to 'timer_sec' for easy plotting in matplotlib
                df_plot = df.set_index('timer_sec')
                
                img_buf = run_analytics.create_infographic(
                    "Interval Session", stats, scores['Total'], "Session Score", 
                    df_plot, 'gap_speed_mps', res['target_mps'], 
                    # PASSING EXTRA ARGS FOR COMPLEX GRAPH
                    intervals_df=df_ints, warm_min=warm, work_min=work, rest_min=rest, reps=reps
                )
                e2.download_button("Download Infographic (PNG)", img_buf, "interval_card.png", "image/png")

        elif mode == "Recovery":
            st.subheader("Recovery Discipline")
            target_rec_hr = st.number_input("Target Max HR", value=145)
            if st.button("Analyze Recovery"):
                res = run_analytics.analyze_recovery(df, target_rec_hr)
                st.session_state['rec_results'] = res

            if st.session_state['rec_results']:
                res = st.session_state['rec_results']
                r1, r2, r3, r4 = st.columns(4)
                r1.metric("Score", f"{res['score']}/100", delta=res['status'])
                r2.metric("Avg HR", f"{res['avg_hr']} bpm")
                r3.metric("Max HR", f"{res['max_hit']} bpm")
                r4.metric("Time > Cap", f"{res['violation_time_min']} min")
                
                fig = run_analytics.create_recovery_figure(res['minute_data'], target_rec_hr)
                st.plotly_chart(fig, use_container_width=True)
                
                st.divider()
                st.subheader("📤 Export Results")
                e1, e2 = st.columns(2)
                
                csv = res['minute_data'].to_csv(index=False).encode('utf-8')
                e1.download_button("Download Minute Data (CSV)", csv, "recovery_stats.csv", "text/csv")
                
                stats = {"Avg HR": f"{res['avg_hr']} bpm", "Max HR": f"{res['max_hit']} bpm", "Time Over": f"{res['violation_time_min']} mins"}
                df_plot = df.set_index('timer_sec')
                img_buf = run_analytics.create_infographic(
                    "Recovery Run", stats, res['score'], "Discipline Score",
                    df_plot, 'hr', target_rec_hr
                )
                e2.download_button("Download Infographic (PNG)", img_buf, "recovery_card.png", "image/png")

        elif mode == "Ghost Battle":
            st.subheader("👻 Ghost Battle Analysis")
            if df_ghost is not None:
                fig = run_analytics.create_ghost_figure(df, df_ghost)
                if fig:
                    st.plotly_chart(fig, use_container_width=True)
                    main_time = df['timer_sec'].max()
                    ghost_time = df_ghost['timer_sec'].max()
                    diff = main_time - ghost_time
                    g1, g2 = st.columns(2)
                    g1.metric("Your Time", f"{int(main_time//60)}:{int(main_time%60):02d}")
                    g2.metric("Ghost Time", f"{int(ghost_time//60)}:{int(ghost_time%60):02d}", delta=f"{-diff:.1f} sec" if diff < 0 else f"+{diff:.1f} sec", delta_color="inverse")
            else:
                st.warning("Please upload a Ghost file in the sidebar to use this mode.")

        else:
            st.subheader("Performance Data")
            st.line_chart(df, x='total_dist_m', y=['speed_smooth', 'hr'])
else:
    st.info("👈 Upload a .tcx/.gpx file or click 'Try with Demo Data' to start.")