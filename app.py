import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import os
import sys
import time
from datetime import date, timedelta

# Add scripts folder to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'scripts'))
from modules import run_analytics, fitness_manager, planner

# --- PAGE CONFIG ---
st.set_page_config(page_title="Run Analysis Tool", layout="wide", page_icon="🏃‍♂️")

st.title("🏃‍♂️ Run Analysis Tool")
st.markdown("Analysis for the **Norwegian Method**.")

# --- SIDEBAR CONFIG ---
st.sidebar.header("Navigation")
tool_mode = st.sidebar.radio("Select Module:", ["Run Analyser", "Pace Calculator", "Fitness Manager", "Guidelines"])

st.sidebar.divider()

# GLOBAL SETTINGS
user_max_hr = st.sidebar.number_input("Your Max Heart Rate", 150, 220, 190)
user_rest_hr = st.sidebar.number_input("Your Resting Heart Rate", 30, 100, 50)

# DATA LOADER
st.sidebar.divider()
st.sidebar.header("Current Activity")
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
    uploaded_file = st.sidebar.file_uploader("Upload Run File (.tcx/.gpx)", type=['tcx', 'gpx'])

# DEFAULT GPS SMOOTHING
smoothing = st.sidebar.slider("GPS Smoothing", 0, 60, 30)

@st.cache_data
def load_data(file_bytes, file_name, smooth_sec):
    safe_name = os.path.basename(file_name) 
    temp_filename = f"temp_{safe_name}"
    with open(temp_filename, "wb") as f:
        f.write(file_bytes)
    try:
        # Returns tuple: (DataFrame, Detected_Laps_List)
        df, laps = run_analytics.parse_file(temp_filename, smoothing_span=smooth_sec)
        return df, laps
    finally:
        if os.path.exists(temp_filename): os.remove(temp_filename)

# SESSION STATE
if 'int_results' not in st.session_state: st.session_state['int_results'] = None
if 'rec_results' not in st.session_state: st.session_state['rec_results'] = None
if 'pace_results' not in st.session_state: st.session_state['pace_results'] = None

# Define Standard Schema
STD_COLS = ['Date', 'Activity', 'Type', 'Load', 'Duration', 'Z1', 'Z2', 'Z3', 'Z4', 'Z5']

if 'fitness_history' not in st.session_state: 
    st.session_state['fitness_history'] = pd.DataFrame(columns=STD_COLS)

if 'loaded_file_id' not in st.session_state:
    st.session_state['loaded_file_id'] = None

# HELPER: Enforce Schema and Sort
def clean_and_sort_history(df):
    for c in STD_COLS:
        if c not in df.columns: df[c] = 0
    df = df[STD_COLS].copy()
    df['Date'] = pd.to_datetime(df['Date']).dt.date
    df = df.sort_values('Date', ascending=False)
    return df

# ==============================================================================
# TOOL 1: GUIDELINES
# ==============================================================================
if tool_mode == "Guidelines":
    st.header("📖 Methodology & Guidelines")
    tab1, tab2, tab3 = st.tabs(["The Method", "App Manual", "Resources & Links"])
    
    with tab1:
        st.markdown("""
        ### The Norwegian Singles Method
        *Adapted from Marius Bakken & "Sirpoc"*
        
        **The Philosophy:**
        Train exactly at the physiological tipping point (Threshold) without crossing it. By staying slightly *below* threshold (Sub-Threshold), you can accumulate massive volume without the fatigue of traditional "Hard" workouts.

        **1. Single vs. Double Threshold**
        * **Elites (Ingebrigtsen):** Run "Double Threshold" (AM and PM sessions) to maximize time-in-zone.
        * **Us (The Method):** We focus on **Single Threshold**. We perform longer, single sessions (e.g., 10x3min or 3x10min) to get the benefit without the lifestyle cost of two showers a day.

        **2. The 75/25 Rule**
        * **75% of Volume:** Strict Zone 1/2. Easy, conversational running.
        * **25% of Volume:** Sub-Threshold Intervals (Zone 3/Low Zone 4).
        * **0% of Volume:** "Grey Zone" (High Z4/Z5). We rarely sprint or race in training.
        """)

    with tab2:
        st.markdown("""
        ### 🛠️ App User Guide

        #### 1. Run Analyser
        Upload your `.tcx` or `.gpx` files to analyse individual sessions.
        * **Intervals Mode:** Analyses "Sub-Threshold" sessions.
            * *New:* Choose **Manual Entry** (define your own structure) or **File Laps** (uses the lap button presses from your watch).
            * *Metrics:* Tracks Pace Discipline (GAP), Heart Rate Compliance, and **Heart Rate Recovery (HRR)** (beats dropped in 60s rest).
        * **Recovery Mode:** Verifies you stayed below your HR cap.
        * **Ghost Battle:** Compare two files against each other.

        #### 2. Pace Calculator & Weekly Architect
        * **Calculator:** Enter a recent race result to get exact training paces.
        * **Weekly Architect:** A smart scheduler that builds a balanced week.
            * Set your total time goal (e.g., 5 hours).
            * Configure your days (Run, Rest, Long Run).
            * The tool calculates optimal interval sessions to hit the **20-25% intensity target** without overtraining.

        #### 3. Fitness Manager
        Long-term trend analysis.
        * **Import:** Supports CSV exports from **Intervals.icu** and **Strava**.
        * **Polarisation:** Visual gauges show your distribution of Easy vs. Hard running (Goal: ~75% Easy).
        * **PMC:** Tracks Fitness (CTL) and Fatigue (ATL) over time.
        """)

    with tab3:
        st.markdown("""
        ### 📚 Recommended Resources

        #### The Book
        **"Norwegian Singles Method: Subthreshold Running Kept Simple"**
        * *Highly recommended.* The definitive guide to applying these principles for non-elite runners. It explains the "Why" behind the "Singles" approach used in this tool.

        #### Useful Links
        * [Threshold.works](https://threshold.works) - Excellent resource for training logic and calculators.
        * [Sub-Threshold Guide](https://sites.google.com/view/sub-threshold/) - Detailed breakdown of the methodology and science.
        * [Marius Bakken's Website](http://www.mariusbakken.com/) - The originator of the modern Norwegian model.

        ### Glossary
        * **Sub-T (Sub-Threshold):** Running slightly slower than your Anaerobic Threshold (approx HM to Marathon pace).
        * **GAP (Grade Adjusted Pace):** Real-time pace adjustment for hills.
        * **HRR (Heart Rate Recovery):** The beats per minute your HR drops 60s after an interval.
        """)

# ==============================================================================
# TOOL 2: PACE CALCULATOR (Updated with Weekly Planner)
# ==============================================================================
elif tool_mode == "Pace Calculator":
    st.subheader("Norwegian Singles Pace Calculator")
    
    # TABS
    ptab1, ptab2 = st.tabs(["Calculator", "Weekly Planner"])
    
    with ptab1:
        st.info("Enter a recent race result to get specific training targets.")
        
        col1, col2 = st.columns(2)
        race_dist = col1.selectbox("Race Distance", ["5k", "10k", "Half Marathon", "Marathon"])
        race_time = col2.text_input("Race Time (HH:MM:SS or MM:SS)", "20:00")
        
        if st.button("Calculate Targets"):
            dist_map = {"5k": 5, "10k": 10, "Half Marathon": 21.1, "Marathon": 42.2}
            base_paces, matrix_df = run_analytics.calculate_training_paces(dist_map[race_dist], race_time)
            
            if base_paces:
                st.session_state['pace_results'] = {
                    'base': base_paces,
                    'matrix': matrix_df,
                    'dist': race_dist,
                    'time': race_time
                }
            else:
                st.error("Invalid time format.")

        # DISPLAY FROM STATE
        if st.session_state['pace_results']:
            res = st.session_state['pace_results']
            base_paces = res['base']
            matrix_df = res['matrix']
            
            st.divider()
            st.markdown("### Neutral Conditions Targets (15°C / Light Wind)")
            c1, c2, c3 = st.columns(3)
            
            short_k = next(k for k in base_paces if "Short" in k)
            med_k = next(k for k in base_paces if "Medium" in k)
            long_k = next(k for k in base_paces if "Long" in k)
            
            c1.metric("Short Intervals", base_paces[short_k], delta="e.g. 10 x 3min (HM Pace)", help="Protocol: 10-12 reps of 3 minutes. Rest 60s.")
            c2.metric("Medium Intervals", base_paces[med_k], delta="e.g. 6 x 6min (30k Pace)", help="Protocol: 4-6 reps of 6-8 minutes. Rest 60-90s.")
            c3.metric("Long Intervals", base_paces[long_k], delta="e.g. 3 x 10min (Mar Pace)", help="Protocol: 3-4 reps of 10 minutes. Rest 60-90s.")
            
            # DOWNLOADS SECTION
            st.divider()
            d1, d2 = st.columns(2)
            
            # 1. Image Download
            card_buf = run_analytics.create_pace_card(res['dist'], res['time'], base_paces)
            d1.download_button("Download Targets Card (PNG)", card_buf, "training_targets.png", "image/png", use_container_width=True)
            
            # 2. Table Download (CSV)
            pace_data = [{"Session": k.split(' (')[0], "Pace Range": v} for k, v in base_paces.items()]
            df_paces = pd.DataFrame(pace_data)
            csv = df_paces.to_csv(index=False).encode('utf-8')
            d2.download_button("Download Table (CSV)", csv, "targets.csv", "text/csv", use_container_width=True)
            
            st.divider()
            st.markdown("### 🌡️ Conditions Adjustment Matrix")
            st.dataframe(matrix_df, use_container_width=True, hide_index=True)
            st.markdown("*Adjust targets if conditions match these scenarios.*")

    with ptab2:
        st.subheader("🗓️ Weekly Schedule Architect")
        
        # Ensure calculator has run first
        if not st.session_state['pace_results']:
            st.warning("Please run the 'Calculator' tab first to establish your baselines.")
        else:
            res = st.session_state['pace_results']
            
            # --- INPUTS ---
            st.markdown("#### Volume & Constraints")
            c1, c2 = st.columns(2)
            total_vol = c1.slider("Total Weekly Volume (Hours)", 3.0, 10.0, 5.0, 0.25)
            
            w1, w2 = c2.columns(2)
            w_up = w1.number_input("Sub-T Warmup (min)", 10, 60, 30)
            c_down = w2.number_input("Sub-T Cool Down (min)", 5, 60, 30)
            
            st.markdown("#### Day Configuration")
            d_cols = st.columns(7)
            days = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
            
            # Defaults: Mon=SubT, Wed=SubT, Sun=Long
            defaults = [3, 1, 3, 1, 0, 1, 2] 
            
            day_config = {}
            has_race = False
            
            for i, day in enumerate(days):
                val = d_cols[i].selectbox(day, ["Rest", "Easy Run", "Long Run", "Sub-T", "Race"], index=defaults[i], key=f"d_{day}")
                day_config[day] = val
                if val == "Race": has_race = True
            
            race_est = 0
            if has_race:
                st.info("🏁 Race detected. Intensity budget will adapt.")
                race_est = st.number_input("Estimated Race Duration (mins)", 15, 300, 45)
            
            # --- VALIDATION ---
            errors, warnings = planner.validate_schedule(day_config)
            
            for err in errors: st.error(err)
            for warn in warnings: st.warning(warn)
            
            if not errors:
                if st.button("Generate Plan"):
                    # Calculate Paces
                    subt_pace, easy_pace = planner.calculate_paces(res['dist'], res['time'])
                    
                    if subt_pace:
                        df_plan = planner.generate_plan(
                            day_config, 
                            total_vol * 60, 
                            race_est, 
                            w_up, 
                            c_down, 
                            subt_pace, 
                            easy_pace
                        )
                        
                        st.divider()
                        st.markdown(f"### Proposed Schedule ({int(total_vol)} Hours)")
                        
                        st.dataframe(
                            df_plan,
                            column_config={
                                "Structure": st.column_config.TextColumn("Session Detail", width="large"),
                                "Duration (min)": st.column_config.ProgressColumn("Volume", format="%d m", min_value=0, max_value=120)
                            },
                            use_container_width=True,
                            hide_index=True
                        )
                        
                        total_m = df_plan['Duration (min)'].sum()
                        st.caption(f"Total Planned Time: {int(total_m // 60)}h {int(total_m % 60)}m")
                        
                        csv = df_plan.to_csv(index=False).encode('utf-8')
                        st.download_button("Download Plan (CSV)", csv, "weekly_plan.csv", "text/csv")
                    else:
                        st.error("Could not calculate paces from base data.")

# ==============================================================================
# TOOL 3: FITNESS MANAGER
# ==============================================================================
elif tool_mode == "Fitness Manager":
    st.subheader("📈 Fitness Manager")
    
    with st.expander("📂 1. Load History (Start Here)", expanded=True):
        hist_file = st.file_uploader("Upload 'training_log.csv' (or Intervals.icu / Strava CSV)", type=['csv'])
        if hist_file is not None:
            if st.session_state['loaded_file_id'] != hist_file.name:
                try:
                    df_hist = pd.read_csv(hist_file)
                    
                    if 'Load' in df_hist.columns and 'Date' in df_hist.columns:
                        st.session_state['fitness_history'] = clean_and_sort_history(df_hist)
                        st.session_state['loaded_file_id'] = hist_file.name
                        st.success(f"Loaded {len(df_hist)} activities.")
                    else:
                        hist_file.seek(0)
                        df_imp, msg = fitness_manager.parse_intervals_csv(hist_file)
                        if df_imp is not None:
                            st.session_state['fitness_history'] = clean_and_sort_history(df_imp)
                            st.session_state['loaded_file_id'] = hist_file.name
                            st.success(f"Imported {len(df_imp)} from Intervals.icu!")
                        else:
                            hist_file.seek(0)
                            df_imp, msg = fitness_manager.parse_strava_csv(hist_file, user_max_hr, user_rest_hr)
                            if df_imp is not None:
                                st.session_state['fitness_history'] = clean_and_sort_history(df_imp)
                                st.session_state['loaded_file_id'] = hist_file.name
                                st.success(f"Imported {len(df_imp)} from Strava.")
                            else:
                                st.error("Could not parse file.")
                except Exception as e:
                    st.error(f"Error loading file: {e}")

    st.divider()
    c1, c2 = st.columns(2)
    
    with c1:
        st.markdown("#### Add from Current Upload")
        if uploaded_file:
            curr_type = st.selectbox("Activity Type", ["Run", "Walk", "Hike", "Cycle", "Swim", "Elliptical", "Strength"], key='upload_type')
            
            if st.button("Calculate & Add to Log"):
                df_curr, _ = load_data(uploaded_file.getvalue(), uploaded_file.name, smoothing)
                if df_curr is not None:
                    duration = int(df_curr['timer_sec'].max() / 60)
                    avg_hr = int(df_curr['hr'].mean()) if 'hr' in df_curr.columns and not df_curr['hr'].isnull().all() else 0
                    
                    if avg_hr > 0:
                        load = fitness_manager.calculate_trimp(duration, avg_hr, user_max_hr, user_rest_hr)
                        zones_df = run_analytics.calculate_zones(df_curr, user_max_hr)
                        z_vals = {'Z1':0, 'Z2':0, 'Z3':0, 'Z4':0, 'Z5':0}
                        if zones_df is not None:
                            for _, r in zones_df.iterrows():
                                if r['Zone'] in z_vals: z_vals[r['Zone']] = int(r['Minutes'])
                        
                        new_row = pd.DataFrame([{
                            'Date': df_curr['time'].iloc[0].date(),
                            'Activity': 'Uploaded Session', 'Type': curr_type,
                            'Load': load, 'Duration': duration,
                            'Z1': z_vals['Z1'], 'Z2': z_vals['Z2'], 'Z3': z_vals['Z3'],
                            'Z4': z_vals['Z4'], 'Z5': z_vals['Z5']
                        }])
                        
                        updated_df = pd.concat([st.session_state['fitness_history'], new_row], ignore_index=True)
                        st.session_state['fitness_history'] = clean_and_sort_history(updated_df)
                        
                        st.toast(f"✅ {curr_type} Added! Score: {load} TSS", icon="🎉")
                        time.sleep(1.5) 
                        st.rerun() 
                    else:
                        st.error("No Heart Rate data found.")
        else:
            st.info("Upload GPX/TCX file in the sidebar to enable this.")

    with c2:
        st.markdown("#### Manual Entry")
        with st.form("manual_add"):
            m_date = st.date_input("Date", date.today())
            m_type = st.selectbox("Type", ["Run", "Gym", "Cycle", "Swim", "Walk", "Hike"])
            m_dur = st.number_input("Duration (min)", 0, 300, 60)
            m_rpe = st.slider("Intensity (RPE 1-10)", 1, 10, 5)
            st.markdown("Optional: Zone Breakdown (min)")
            zc1, zc2 = st.columns(2)
            z1_in = zc1.number_input("Z1/Z2 (Easy)", value=m_dur)
            z3_in = zc2.number_input("Z3+ (Hard)", value=0)
            
            if st.form_submit_button("Add Activity"):
                est_load = int(m_dur * (m_rpe * 10) / 60)
                new_row = pd.DataFrame([{
                    'Date': m_date, 'Activity': 'Manual', 'Type': m_type,
                    'Load': est_load, 'Duration': m_dur,
                    'Z1': z1_in, 'Z2': 0, 'Z3': z3_in, 'Z4': 0, 'Z5': 0 
                }])
                updated_df = pd.concat([st.session_state['fitness_history'], new_row], ignore_index=True)
                st.session_state['fitness_history'] = clean_and_sort_history(updated_df)
                
                st.toast(f"✅ Manual Entry Added!", icon="✍️")
                time.sleep(1.5)
                st.rerun() 

    st.divider()
    df_hist = st.session_state['fitness_history']
    
    if not df_hist.empty:
        df_calc = df_hist.sort_values('Date', ascending=True)
        df_pmc = fitness_manager.calculate_pmc(df_calc)
        today_stats = df_pmc.iloc[-1]
        
        m1, m2, m3 = st.columns(3)
        m1.metric("Fitness (CTL)", f"{int(today_stats['CTL'])}", delta="42 Day Avg")
        m2.metric("Fatigue (ATL)", f"{int(today_stats['ATL'])}", delta="7 Day Avg", delta_color="inverse")
        m3.metric("Form (TSB)", f"{int(today_stats['TSB'])}", delta="Balance")
        
        st.markdown("### 📊 Training Distribution (Last 30 Days)")
        
        cutoff_date = date.today() - timedelta(days=30)
        df_recent = df_calc[df_calc['Date'] >= cutoff_date]
        
        if not df_recent.empty:
            t_z12 = df_recent['Z1'].sum() + df_recent['Z2'].sum()
            t_hard = df_recent['Z3'].sum() + df_recent['Z4'].sum() + df_recent['Z5'].sum()
            t_total = t_z12 + t_hard
            
            df_run = df_recent[df_recent['Type'] == 'Run']
            r_z12 = df_run['Z1'].sum() + df_run['Z2'].sum()
            r_hard = df_run['Z3'].sum() + df_run['Z4'].sum() + df_run['Z5'].sum()
            r_total = r_z12 + r_hard

            c1, c2 = st.columns(2)
            
            with c1:
                st.markdown("#### 🌐 All Activities")
                if t_total > 0:
                    pt_easy = int((t_z12 / t_total) * 100)
                    pt_hard = 100 - pt_easy
                    st.progress(pt_easy/100, text=f"{pt_easy}% Easy / {pt_hard}% Hard")
                else:
                    st.info("No data in 30 days.")

            with c2:
                st.markdown("#### 🏃 Running Only")
                if r_total > 0:
                    pr_easy = int((r_z12 / r_total) * 100)
                    pr_hard = 100 - pr_easy
                    st.progress(pr_easy/100, text=f"{pr_easy}% Easy / {pr_hard}% Hard")
                else:
                    st.info("No runs in 30 days.")
        else:
            st.info("No activity data found for the last 30 days.")
        
        st.plotly_chart(fitness_manager.create_pmc_figure(df_pmc), use_container_width=True)
        st.plotly_chart(fitness_manager.create_zone_figure(df_calc), use_container_width=True)
        
        st.markdown("### 💾 Save Your Data")
        csv = df_hist.to_csv(index=False).encode('utf-8')
        st.download_button("Download Updated Log", csv, "training_log.csv", "text/csv", key='down_csv')
        
        with st.expander("View Raw Data (Most Recent First)", expanded=True):
            st.dataframe(df_hist, use_container_width=True)
    else:
        st.info("No data loaded.")

# ==============================================================================
# TOOL 4: RUN ANALYZER
# ==============================================================================
elif tool_mode == "Run Analyser":
    if uploaded_file is not None:
        file_bytes = uploaded_file.read() if hasattr(uploaded_file, 'read') else uploaded_file.getvalue()
        file_name = getattr(uploaded_file, 'name', 'demo_file.tcx')

        with st.spinner("Processing..."):
            df, detected_laps = load_data(file_bytes, file_name, smoothing)

        st.sidebar.divider()
        ghost_file_upload = st.sidebar.file_uploader("Optional: Ghost File", type=['tcx', 'gpx'])
        df_ghost = None
        if ghost_file_upload:
            with st.spinner("Processing Ghost..."):
                df_ghost, _ = load_data(ghost_file_upload.getvalue(), ghost_file_upload.name, smoothing)

        if df is not None:
            has_dist = df['total_dist_m'].max() > 100
            
            run_date = df['time'].iloc[0].strftime('%Y-%m-%d %H:%M')
            duration_min = df['timer_sec'].max() / 60
            
            if has_dist:
                dist_km = df['total_dist_m'].max() / 1000
                decoupling = run_analytics.calculate_decoupling(df)
            else:
                dist_km = 0
                decoupling = None
            
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Date", run_date)
            c2.metric("Distance", f"{dist_km:.2f} km")
            c3.metric("Duration", f"{int(duration_min)} mins")
            c4.metric("Decoupling", f"{decoupling}%" if decoupling else "N/A", delta="Good" if decoupling and decoupling < 5 else "Drift" if decoupling else None, delta_color="inverse")
            st.divider()

            if not has_dist:
                st.warning("⚠️ No GPS distance found. Speed/Pace metrics are unavailable. (Interval Mode Disabled)")
                modes = ["Recovery"]
            else:
                modes = ["Select Analysis Mode", "Standard", "Intervals", "Recovery"]
                if df_ghost: modes.append("Ghost Battle")
                
            mode = st.radio("Mode:", modes, horizontal=True)

            if mode == "Intervals" and has_dist:
                st.subheader("Norwegian Singles Analysis")
                
                det_mode = st.radio("Interval Definition:", ["Manual Entry", "Use File Laps (Watch Button)"], horizontal=True)
                
                with st.form("interval_config"):
                    if det_mode == "Manual Entry":
                        c1, c2, c3, c4 = st.columns(4)
                        warm = c1.number_input("Warmup (min)", value=15)
                        reps = c2.number_input("Reps", value=10, step=1)
                        work = c3.number_input("Work (min)", value=3.0)
                        rest = c4.number_input("Rest (min)", value=1.0)
                        cool = 10 
                        laps_to_use = None
                    else:
                        if len(detected_laps) > 0:
                            st.info(f"Found {len(detected_laps)} laps in file.")
                            laps_to_use = detected_laps
                            warm, reps, work, rest, cool = 0, 0, 0, 0, 0
                        else:
                            st.error("No Laps found in file. Switch to Manual.")
                            laps_to_use = None
                            warm, reps, work, rest, cool = 0, 0, 0, 0, 0

                    c5, c6, c7 = st.columns(3)
                    if det_mode == "Manual Entry":
                        cool = c5.number_input("Cool (min)", value=10)
                    
                    buffer = c6.number_input("Buffer (sec)", value=30)
                    target_pace = c7.text_input("Target Pace", value="3:45")
                    target_hr = c5.number_input("HR Cap", value=170)
                    
                    e1, e2, e3 = st.columns(3)
                    temp_c = e1.number_input("Temperature (°C)", value=15, step=1)
                    wind_kmh = e2.number_input("Wind Speed (km/h)", value=0, step=5)
                    surface_pen = e3.number_input("Surface Penalty (sec/km)", value=0, step=1)
                    
                    o1, o2, o3 = st.columns(3)
                    use_gap = o1.checkbox("Use GAP", value=True)
                    use_hr = o2.checkbox("Include HR", value=True)
                    lenient_rep1 = o3.checkbox("Warmup Rep 1", value=True)
                    
                    if st.form_submit_button("Run Analysis"):
                        df_ints, target_mps, scores = run_analytics.analyze_intervals(
                            df, warm, work, rest, reps, cool, buffer, target_pace, target_hr, 
                            use_gap, not use_hr, lenient_rep1, temp_c, wind_kmh, surface_pen,
                            custom_laps=laps_to_use
                        )
                        st.session_state['int_results'] = {'df_ints': df_ints, 'target_mps': target_mps, 'scores': scores, 'params': (target_pace, warm, work, rest, reps, target_hr, use_gap, use_hr)}

                if st.session_state['int_results']:
                    res = st.session_state['int_results']
                    scores = res['scores']
                    
                    s1, s2, s3 = st.columns(3)
                    s1.metric("Session Score", f"{scores['Total']}/100", delta="Target: 100")
                    s2.metric("Pace Discipline", f"{scores['Pace Pts']}/50")
                    s3.metric("HR Compliance", f"{scores['HR Pts']}/50")

                    df_display = res['df_ints'].drop(columns=['HRR (60s)', 'Start Sec', 'End Sec'], errors='ignore')
                    st.dataframe(df_display.set_index('Rep'), use_container_width=True)
                    
                    csv = res['df_ints'].to_csv(index=False).encode('utf-8')
                    st.download_button("Download CSV Data", csv, "interval_data.csv", "text/csv")
                    
                    fig = run_analytics.create_interval_figure(df, res['df_ints'], res['params'][0], res['target_mps'], res['params'][1], res['params'][2], res['params'][3], res['params'][4], res['params'][6])
                    if fig: st.plotly_chart(fig, use_container_width=True)
                    
                    st.divider()
                    fig_hrr = run_analytics.create_hrr_figure(res['df_ints'])
                    if fig_hrr: 
                        st.plotly_chart(fig_hrr, use_container_width=True)
                    else:
                        st.info("No Heart Rate Recovery data available (check rest duration >= 1min).")

                    stats = {"Reps": f"{len(res['df_ints'])} Reps", "Target Pace": target_pace}
                    img_buf = run_analytics.create_infographic("Interval Session", stats, scores['Total'], "Session Score", df.set_index('timer_sec'), 'gap_speed_mps' if use_gap else 'speed_smooth', res['target_mps'], intervals_df=res['df_ints'], warm_min=warm, work_min=work, rest_min=rest, reps=reps, use_gap=use_gap)
                    st.download_button("Download Infographic", img_buf, "interval_card.png", "image/png")

            elif mode == "Recovery":
                target_rec_hr = st.number_input("Target Max HR", value=145)
                if st.button("Analyze Recovery"):
                    st.session_state['rec_results'] = run_analytics.analyze_recovery(df, target_rec_hr)
                
                if st.session_state['rec_results']:
                    res = st.session_state['rec_results']
                    r1, r2 = st.columns(2)
                    r1.metric("Score", f"{res['score']}/100", delta=res['status'])
                    st.plotly_chart(run_analytics.create_recovery_figure(res['minute_data'], target_rec_hr), use_container_width=True)
                    
                    rec_buf = run_analytics.create_recovery_card(res, res['score'], res['status'])
                    st.download_button("Download Recovery Card", rec_buf, "recovery_card.png", "image/png")
                    
                    if has_dist:
                        run_analytics.generate_dashboard(df, "temp_dash.html", "Recovery Deep Dive", user_max_hr)
                        with open("temp_dash.html", 'r', encoding='utf-8') as f:
                            html_data = f.read()
                        st.components.v1.html(html_data, height=1000, scrolling=True)

            elif mode == "Ghost Battle":
                if df_ghost is not None:
                    fig = run_analytics.create_ghost_figure(df, df_ghost)
                    st.plotly_chart(fig, use_container_width=True)

            elif mode == "Standard" and has_dist:
                run_analytics.generate_dashboard(df, "temp_dash.html", "Standard Run Analysis", user_max_hr)
                with open("temp_dash.html", 'r', encoding='utf-8') as f:
                    html_data = f.read()
                st.components.v1.html(html_data, height=1000, scrolling=True)
                
    else:
        st.info("👈 Upload a .tcx/.gpx file or click 'Try with Demo Data' to start.")