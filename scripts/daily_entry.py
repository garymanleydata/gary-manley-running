import sqlite3
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
from datetime import datetime

# 1. Setup Configuration & Paths
# ---------------------------------------------------------
vScriptDir = os.path.dirname(os.path.abspath(__file__))
vDBPath = os.path.join(vScriptDir, '../data/journal.db')
vImagesPath = os.path.join(vScriptDir, '../assets/images')
vPostsPath = os.path.join(vScriptDir, '../posts')

os.makedirs(os.path.dirname(vDBPath), exist_ok=True)
os.makedirs(vImagesPath, exist_ok=True)
os.makedirs(vPostsPath, exist_ok=True)

vCategories = [
    'sleep', 'cardio_1', 'cardio_2', 'cardio_3', 
    'prehab', 'strength_training', 'eating_window', 
    'mental_training', 'visualisation'
]

# 2. Database Functions
# ---------------------------------------------------------
def init_db():
    vConn = sqlite3.connect(vDBPath)
    vCursor = vConn.cursor()
    vCols = ", ".join([f"{c} INTEGER" for c in vCategories])
    vQuery = f"""
        CREATE TABLE IF NOT EXISTS daily_journal (
            date TEXT PRIMARY KEY,
            {vCols},
            notes TEXT
        )
    """
    vCursor.execute(vQuery)
    vConn.commit()
    vConn.close()

def save_to_db(vDate, vScores):
    vConn = sqlite3.connect(vDBPath)
    vCursor = vConn.cursor()
    vPlaceholders = ", ".join(["?" for _ in vCategories])
    vColNames = ", ".join(vCategories)
    vQuery = f"""
        INSERT OR REPLACE INTO daily_journal (date, {vColNames})
        VALUES (?, {vPlaceholders})
    """
    vData = [vDate] + vScores
    vCursor.execute(vQuery, vData)
    vConn.commit()
    vConn.close()
    print(f"Data saved to {vDBPath}")

# 3. Visualization & Narrative Functions
# ---------------------------------------------------------
def generate_radar_chart(vDate, vScores):
    vN = len(vCategories)
    vAngles = [n / float(vN) * 2 * np.pi for n in range(vN)]
    vAngles += vAngles[:1]
    vValues = vScores + vScores[:1]
    
    fig, ax = plt.subplots(figsize=(6, 6), subplot_kw=dict(polar=True))
    plt.xticks(vAngles[:-1], vCategories, color='grey', size=8)
    ax.set_rlabel_position(0)
    plt.yticks([2, 4, 6, 8, 10], ["2", "4", "6", "8", "10"], color="grey", size=7)
    plt.ylim(0, 10)
    ax.plot(vAngles, vValues, linewidth=2, linestyle='solid', color='#1f77b4')
    ax.fill(vAngles, vValues, 'b', alpha=0.1)
    
    vFilename = f"{vDate}_radar.png"
    vFullPath = os.path.join(vImagesPath, vFilename)
    plt.savefig(vFullPath)
    plt.close()
    return vFilename

def generate_narrative(vScores):
    """Analyzes scores to produce a text summary."""
    # Create a dictionary for easier sorting
    vData = dict(zip(vCategories, vScores))
    
    # Filter based on thresholds
    vHighs = [k.replace('_', ' ').title() for k, v in vData.items() if v >= 8]
    vLows = [k.replace('_', ' ').title() for k, v in vData.items() if v <= 4]
    
    vText = ""
    
    if vHighs:
        vText += f"**Wins:** Strong performance today in {', '.join(vHighs)}. "
    
    if vLows:
        vText += f"**Focus Areas:** Need to give more attention to {', '.join(vLows)}. "
        
    if not vHighs and not vLows:
        vText += "A balanced day with no major outliers."
        
    return vText

# 4. Post Creation Function
# ---------------------------------------------------------
def create_quarto_post(vDate, vImageFilename, vNarrative):
    vPostFilename = os.path.join(vPostsPath, f"{vDate}-journal.qmd")
    vRelativeImgPath = f"../assets/images/{vImageFilename}"
    
    vTemplate = f"""---
title: "Journal: {vDate}"
date: "{vDate}"
categories: [journal, data]
image: {vRelativeImgPath}
---

## Daily Snapshot

![]({vRelativeImgPath})

> {vNarrative}

## Notes

*Enter your written journal notes here...*

"""
    with open(vPostFilename, 'w') as f:
        f.write(vTemplate)
    print(f"Post created: {vPostFilename}")

# 5. Main Execution
# ---------------------------------------------------------
if __name__ == "__main__":
    init_db()
    vDateToday = datetime.now().strftime("%Y-%m-%d")
    print(f"\n--- Journal Entry for {vDateToday} ---")
    
    vDailyScores = []
    for vCat in vCategories:
        while True:
            try:
                vInput = input(f"Score for '{vCat}' (0-10): ")
                vScore = int(vInput)
                if 0 <= vScore <= 10:
                    vDailyScores.append(vScore)
                    break
                else:
                    print("Please enter 0-10.")
            except ValueError:
                print("Invalid number.")

    save_to_db(vDateToday, vDailyScores)
    vImgFile = generate_radar_chart(vDateToday, vDailyScores)
    vNarrativeText = generate_narrative(vDailyScores) # Generate the text
    create_quarto_post(vDateToday, vImgFile, vNarrativeText) # Pass it to the template
    
    print("\nDone! Narrative and Graph generated.")