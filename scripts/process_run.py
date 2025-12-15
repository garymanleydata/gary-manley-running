import os
import sys

# --- PATH FIX ---
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)
# ----------------

from modules import run_analytics 
from datetime import datetime

# CONFIG
vDataDir = os.path.join(os.path.dirname(__file__), '../data/runs')
vPostDir = os.path.join(os.path.dirname(__file__), '../posts')
vAssetDir = os.path.join(os.path.dirname(__file__), '../assets')

def select_file():
    files = [f for f in os.listdir(vDataDir) if f.lower().endswith(('.tcx', '.gpx'))]
    if not files:
        print("Error: No .tcx or .gpx files found")
        return None
    full_paths = [os.path.join(vDataDir, f) for f in files]
    latest_file = max(full_paths, key=os.path.getctime)
    latest_filename = os.path.basename(latest_file)

    print(f"\nLatest file detected: [ {latest_filename} ]")
    user_input = input("Press ENTER to process, or type filename: ").strip()
    
    if user_input == "": return os.path.join(vDataDir, latest_filename)
    else:
        manual_path = os.path.join(vDataDir, user_input)
        return manual_path if os.path.exists(manual_path) else None

def main():
    print("--- RUN PROCESSING ENGINE ---")
    
    target_file = select_file()
    if not target_file: return
    
    smoothing_input = input("GPS Smoothing Window (seconds) [Default 15]: ")
    smooth_sec = int(smoothing_input) if smoothing_input else 15
    
    df = run_analytics.parse_file(target_file, smooth_sec)
    if df is None: return
    
    vDate = df['time'].iloc[0].strftime('%Y-%m-%d')
    vDateStr = vDate 
    
    # 1. Map Question
    map_yn = input("Generate Route Map? (y/n) [Default n]: ").strip().lower()
    map_file = f"map_{vDateStr}.html"
    map_html_block = ""
    
    if map_yn == 'y':
        print("Generating Map...")
        run_analytics.generate_map(df, os.path.join(vAssetDir, 'maps', map_file))
        map_html_block = f"""
## Route
<iframe src="../assets/maps/{map_file}" width="100%" height="400" style="border:none;"></iframe>
"""
    
    print("Generating Dashboard...")
    dash_file = f"dashboard_{vDateStr}.html"
    run_analytics.generate_dashboard(df, os.path.join(vAssetDir, 'graphs', dash_file))
    
    print("\nSelect Session Type:")
    print("1. Standard / Parkrun")
    print("2. Recovery Run")
    print("3. Norwegian Singles (Intervals)")
    
    choice = input("Enter choice (1-3): ").strip()
    post_content = ""
    title = f"Run: {vDate}"
    
    # === PATH 3: INTERVALS ===
    if choice == '3':
        title = f"Norwegian Singles: {vDate}"
        print("\nInterval Config (Press Enter for defaults)")
        
        def get_input(prompt, default):
            val = input(f"{prompt} (Default {default}): ")
            return float(val) if val else default
            
        def get_str_input(prompt, default):
            val = input(f"{prompt} (Default {default}): ")
            return val if val else default

        warm = get_input("Warmup (mins)", 15)
        reps = int(get_input("Number of Reps", 10))
        work = get_input("Work Duration (mins)", 3)
        rest = get_input("Rest Duration (mins)", 1)
        cool = get_input("Cooldown (mins)", 10)
        buffer = get_input("Buffer Window (seconds)", 15)
        
        target_pace = get_str_input("Target Pace (MM:SS)", "3:45")
        target_hr = int(get_input("Target Max HR cap", 170))
        
        # Call Engine (returns 3 items now)
        df_ints, target_mps, scores = run_analytics.analyze_intervals(
            df, warm, work, rest, reps, cool, buffer, target_pace, target_hr
        )
        
        disc_file = f"discipline_{vDateStr}.html"
        run_analytics.generate_interval_graph(
            df, df_ints, os.path.join(vAssetDir, 'graphs', disc_file),
            target_pace, target_mps, warm, work, rest, reps
        )
        
        tbl_name = f"intervals_{vDateStr}.md"
        df_ints.to_markdown(os.path.join(vAssetDir, 'tables', tbl_name), index=False)
        
        # Color the score
        score = scores['Total']
        score_color = "green" if score >= 80 else "orange" if score >= 50 else "red"
        
        post_content = f"""
## Session Score: <span style="color:{score_color}">{score}/100</span>
* **Pace Score:** {scores['Pace Pts']}/50 (Avg pace vs Target +/- 5s, 10s, 15s)
* **HR Score:** {scores['HR Pts']}/50 (Avg {scores['Avg HR Compliance']}% compliance)

**Target:** {target_pace}/km | **HR Cap:** {target_hr} bpm  

{{{{< include ../assets/tables/{tbl_name} >}}}}

### Pace Discipline Graph
Green Band = +/- 10s. Blue Line = **Rep Average**.  
<iframe src="../assets/graphs/{disc_file}" width="100%" height="600" style="border:none;"></iframe>
"""

    elif choice == '2':
        # Recovery logic ...
        title = f"Recovery: {vDate}"
        target_hr = int(input("Target Max HR cap: ") or 145)
        res = run_analytics.analyze_recovery(df, target_hr)
        color = "green" if res['status'] == "SUCCESS" else "red"
        post_content = f"""
## Recovery Check
* **Result:** <span style="color:{color}">{res['status']}</span>
* **Max HR Hit:** {res['max_hit']} bpm
* **Time Violation:** {res['violation_time_min']} mins
"""
    else:
        post_content = "Standard Run Analysis."

    # WRITE POST
    full_path = os.path.join(vPostDir, f"{vDateStr}-run.qmd")
    template = f"""---
title: "{title}"
date: "{vDate}"
categories: [running, data]
format: html
---

{map_html_block}

{post_content}

## Deep Dive Dashboard
<iframe src="../assets/graphs/{dash_file}" width="100%" height="600" style="border:none;"></iframe>
"""
    with open(full_path, 'w') as f:
        f.write(template)
        
    print(f"\nSUCCESS: Post created at {full_path}")

if __name__ == "__main__":
    main()