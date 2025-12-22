import pandas as pd
import numpy as np
import random

# --- SESSION LIBRARY (Categorized by Rep Duration) ---
# Format: (Reps, Duration_Min, Rest_Min, Description)
# Short: Reps 2-4 mins
# Medium: Reps 5-9 mins
# Long: Reps 10+ mins

SESSION_DB = {
    "Short": [
        # Low Vol
        (6, 3, 1.0, "6 x 3min (1min rest)"),
        (8, 3, 1.0, "8 x 3min (1min rest)"),
        (5, 4, 1.5, "5 x 4min (90s rest)"),
        # Med Vol
        (10, 3, 1.0, "10 x 3min (1min rest)"),
        (6, 4, 1.5, "6 x 4min (90s rest)"),
        # High Vol
        (12, 3, 1.0, "12 x 3min (1min rest)"),
        (15, 3, 1.0, "15 x 3min (1min rest)"),
        (8, 4, 1.5, "8 x 4min (90s rest)")
    ],
    "Medium": [
        # Low Vol
        (4, 5, 2.0, "4 x 5min (2min rest)"),
        (4, 6, 2.0, "4 x 6min (2min rest)"),
        # Med Vol
        (5, 5, 2.0, "5 x 5min (2min rest)"),
        (6, 5, 2.0, "6 x 5min (2min rest)"),
        (5, 6, 2.0, "5 x 6min (2min rest)"),
        # High Vol
        (7, 5, 2.0, "7 x 5min (2min rest)"),
        (8, 5, 2.0, "8 x 5min (2min rest)"),
        (6, 6, 2.0, "6 x 6min (2min rest)"),
        (4, 8, 2.0, "4 x 8min (2min rest)")
    ],
    "Long": [
        # Low Vol
        (2, 10, 2.0, "2 x 10min (2min rest)"),
        (3, 10, 2.0, "3 x 10min (2min rest)"),
        # Med Vol
        (4, 10, 2.0, "4 x 10min (2min rest)"),
        (3, 12, 2.0, "3 x 12min (2min rest)"),
        # High Vol
        (5, 10, 2.0, "5 x 10min (2min rest)"),
        (3, 15, 3.0, "3 x 15min (3min rest)"),
        (4, 15, 3.0, "4 x 15min (3min rest)")
    ]
}

def get_best_session_in_category(category, target_work_mins, exclude_names=[]):
    """
    Looks only inside a specific category (Short/Med/Long)
    to find the session matching the target volume.
    """
    best_sess = None
    min_diff = 999
    
    pool = SESSION_DB.get(category, [])
    
    # Filter available
    available = [s for s in pool if s[3] not in exclude_names]
    if not available: available = pool
        
    for sess in available:
        reps, dur, rest, name = sess
        work_total = reps * dur
        diff = abs(target_work_mins - work_total)
        
        if diff < min_diff:
            min_diff = diff
            best_sess = sess
            
    return best_sess

def validate_schedule(days_config):
    warnings = []
    errors = []
    
    day_order = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
    types = [days_config.get(d, 'Rest') for d in day_order]
    
    c_long = types.count("Long Run")
    c_int = types.count("Sub-T") + types.count("Race")
    c_rest = types.count("Rest")
    has_race = types.count("Race") > 0
    
    if c_int > 3:
        errors.append("❌ Too much intensity: Max 3 hard sessions (Sub-T + Race) per week.")
    
    if c_long > 2:
        errors.append("❌ Too many Long Runs: Max 2 allowed.")
    elif c_long == 2:
        warnings.append("⚠️ Warning: 2 Long Runs selected. Ensure you recover well.")
    elif c_long == 0 and not has_race:
        warnings.append("⚠️ Tip: A Long Run is recommended for aerobic base.")
        
    if c_rest > 1:
        warnings.append("ℹ️ Note: The Norwegian model typically favors active recovery (Easy Runs) over full Rest.")
        
    for i in range(len(types)):
        current = types[i]
        next_d = types[(i + 1) % 7]
        is_hard_curr = current in ["Sub-T", "Race"]
        is_hard_next = next_d in ["Sub-T", "Race"]
        
        if is_hard_curr and is_hard_next:
            d1 = day_order[i]
            d2 = day_order[(i + 1) % 7]
            errors.append(f"❌ Back-to-Back Intensity detected ({d1} & {d2}). Insert an Easy or Rest day.")

    return errors, warnings

def calculate_paces(race_dist, race_time_str):
    try:
        parts = list(map(int, race_time_str.split(':')))
        if len(parts) == 2: seconds = parts[0]*60 + parts[1]
        elif len(parts) == 3: seconds = parts[0]*3600 + parts[1]*60 + parts[2]
        else: return None, None
    except: return None, None
    
    dist_m = 5000 if "5k" in race_dist else 10000 if "10k" in race_dist else 21097 if "Half" in race_dist else 42195
    t_hm = seconds * (21097 / dist_m)**1.06
    hm_pace_sec = t_hm / 21.097 
    
    subt_min = int(hm_pace_sec // 60)
    subt_sec = int(hm_pace_sec % 60)
    subt_str = f"{subt_min}:{subt_sec:02d}"
    
    # Easy Pace: 1.50 multiplier (e.g., 3:50 -> 5:45)
    easy_sec = hm_pace_sec * 1.50
    e_min = int(easy_sec // 60)
    e_sec = int(easy_sec % 60)
    easy_str = f"{e_min}:{e_sec:02d}"
    
    return subt_str, easy_str

def generate_plan(days_config, total_mins, race_time_est_min, warm_min, cool_min, subt_pace, easy_pace):
    day_order = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
    
    # Target Intensity: 22% of total volume
    intensity_budget = total_mins * 0.22  
    
    used_sessions = []
    
    types = [days_config.get(d) for d in day_order]
    count_race = types.count("Race")
    count_subt = types.count("Sub-T")
    count_long = types.count("Long Run")
    count_easy = types.count("Easy Run")
    
    # --- TRACKING USED TIME ---
    minutes_scheduled = 0
    subt_configs = {}
    
    # 1. RACES
    if count_race > 0:
        race_work = race_time_est_min * count_race 
        race_total_time = (warm_min + race_time_est_min + 15) * count_race
        minutes_scheduled += race_total_time
        intensity_budget -= race_work

    # 2. SUB-T SESSIONS (Priority Logic)
    if count_subt > 0:
        target_work_per_sess = intensity_budget / count_subt
        # Hard cap for singles
        if target_work_per_sess > 50: target_work_per_sess = 50 
        # Floor for low volume
        if target_work_per_sess < 15: target_work_per_sess = 15

        for i in range(count_subt):
            # PRIORITY LOGIC:
            # i=0 (1st sess) -> Short Reps
            # i=1 (2nd sess) -> Medium Reps
            # i=2 (3rd sess) -> Long Reps
            if i == 0: cat = "Short"
            elif i == 1: cat = "Medium"
            else: cat = "Long"
            
            # Find best volume match within that specific category
            sess = get_best_session_in_category(cat, target_work_per_sess, used_sessions)
            used_sessions.append(sess[3]) 
            
            reps, dur, rest, name = sess
            total_sess_time = warm_min + (reps*dur) + ((reps-1)*rest) + cool_min
            
            subt_configs[i] = {
                'desc': f"**{name}**",
                'detail': f"{warm_min}m Warmup, {name}, {cool_min}m Cool",
                'pace': subt_pace,
                'total_time': int(total_sess_time)
            }
            minutes_scheduled += int(total_sess_time)

    # 3. CALCULATE REMAINING AEROBIC TIME
    remaining_mins = total_mins - minutes_scheduled
    if remaining_mins < 0: remaining_mins = 0
    
    long_run_time = 0
    easy_run_time = 0
    
    if count_long + count_easy > 0:
        # Weighting: Long Run = 2 shares, Easy Run = 1 share
        shares = (count_long * 2.0) + count_easy
        
        minutes_per_share = remaining_mins / shares
        
        if count_long > 0:
            long_run_time = int(minutes_per_share * 2.0)
            # Cap LR at 2h30 (150m) unless explicitly high volume
            if long_run_time > 150:
                excess = long_run_time - 150
                long_run_time = 150
                # Give excess back to easy runs
                if count_easy > 0:
                    minutes_per_share += (excess / count_easy)
        
        if count_easy > 0:
            # We recalculate easy to ensure sum is perfect
            remaining_after_long = remaining_mins - (long_run_time * count_long)
            easy_run_time = int(remaining_after_long / count_easy)
            
            # Floor check
            if easy_run_time < 20: easy_run_time = 20

    # --- BUILD TABLE ---
    plan = []
    subt_counter = 0
    
    for i, day in enumerate(day_order):
        d_type = days_config.get(day)
        
        is_pre_race = False
        next_day_type = days_config.get(day_order[(i+1)%7])
        if next_day_type == "Race": is_pre_race = True
        
        row = {
            "Day": day,
            "Type": d_type,
            "Structure": "-",
            "Paces": "-",
            "Duration (min)": 0
        }
        
        if d_type == "Rest":
            row["Structure"] = "Rest / Mobility"
            
        elif d_type == "Race":
            row["Structure"] = f"{warm_min}m Warmup + Race Effort"
            row["Paces"] = "Max Effort"
            row["Duration (min)"] = int(warm_min + race_time_est_min + 15)
            
        elif d_type == "Sub-T":
            if subt_counter in subt_configs:
                cfg = subt_configs[subt_counter]
                row["Structure"] = cfg['detail']
                row["Paces"] = f"Work: {cfg['pace']}/km"
                row["Duration (min)"] = cfg['total_time']
                subt_counter += 1
                
        elif d_type == "Long Run":
            row["Structure"] = f"{long_run_time} mins Steady Zone 1/2"
            row["Paces"] = f"{easy_pace}/km or slower"
            row["Duration (min)"] = long_run_time
            
        elif d_type == "Easy Run":
            duration = easy_run_time
            note = "Zone 1/2"
            if is_pre_race: 
                duration = int(duration * 0.6)
                note = "Shakeout (Pre-Race)"
            
            row["Structure"] = f"{duration} mins {note}"
            row["Paces"] = f"{easy_pace}/km or slower"
            row["Duration (min)"] = duration
            
        plan.append(row)
        
    return pd.DataFrame(plan)