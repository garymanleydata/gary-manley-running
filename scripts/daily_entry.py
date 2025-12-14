import sqlite3
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
import sys
from datetime import datetime
import google.generativeai as genai
from dotenv import load_dotenv

# 1. Setup Configuration & Paths
# ---------------------------------------------------------
vScriptDir = os.path.dirname(os.path.abspath(__file__))
vDBPath = os.path.join(vScriptDir, '../data/journal.db')
vImagesPath = os.path.join(vScriptDir, '../assets/images')
vPostsPath = os.path.join(vScriptDir, '../posts')

# Load the secret .env file
load_dotenv(os.path.join(vScriptDir, '../.env'))

# Configure Gemini
vAPIKey = os.getenv("GEMINI_API_KEY")
if vAPIKey:
    genai.configure(api_key=vAPIKey)
else:
    print("WARNING: No API Key found in .env file. AI narrative will be disabled.")

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

# 3. Visualization Function
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

# 4. AI Narrative Function
# ---------------------------------------------------------
def get_ai_narrative(vDate, vScores):
    if not vAPIKey:
        return "AI Narrative unavailable (No API Key)."

    vDataStr = ", ".join([f"{cat}: {score}" for cat, score in zip(vCategories, vScores)])
    
    # Updated model to your preferred choice
    vModelName = 'gemini-2.5-flash-lite' 

    vPrompt = f"""
    Act as a high-performance athletic coach. 
    Here are my tracking scores for {vDate} (out of 10):
    {vDataStr}
    
    Write a concise, 2-3 sentence summary of my day. 
    - If scores are high, be encouraging.
    - If scores are low (especially sleep or recovery), be strict but helpful.
    - Do not use markdown headers or bullet points, just a paragraph.
    """

    try:
        model = genai.GenerativeModel(vModelName)
        response = model.generate_content(vPrompt)
        return response.text
    except Exception as e:
        print(f"AI Error ({vModelName}): {e}")
        return "Could not generate AI narrative today."

# 5. Post Creation Function
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

> **Coach's Notes:** {vNarrative}

## Notes

*Enter your written journal notes here...*

"""
    with open(vPostFilename, 'w') as f:
        f.write(vTemplate)
    print(f"Post created: {vPostFilename}")

# 6. Helper: Date Parsing
# ---------------------------------------------------------
def get_target_date():
    """Asks user for date, defaults to Today, supports YYYYMMDD."""
    while True:
        vInput = input(f"\nDate (YYYYMMDD) [Press Enter for Today]: ").strip()
        
        # Option A: User just hit Enter (Today)
        if not vInput:
            return datetime.now().strftime("%Y-%m-%d")
        
        # Option B: User typed "20251225"
        if len(vInput) == 8 and vInput.isdigit():
            try:
                # Convert to datetime then back to string with hyphens
                dt = datetime.strptime(vInput, "%Y%m%d")
                return dt.strftime("%Y-%m-%d")
            except ValueError:
                print("Invalid date. Please use YYYYMMDD (e.g., 20251225).")
                continue
        
        # Option C: User typed "2025-12-25"
        if "-" in vInput:
             try:
                dt = datetime.strptime(vInput, "%Y-%m-%d")
                return dt.strftime("%Y-%m-%d")
             except ValueError:
                pass
        
        print("Invalid format. Try 20251225.")

# 7. Main Execution
# ---------------------------------------------------------
if __name__ == "__main__":
    init_db()
    
    # STEP 1: Determine the Date
    vTargetDate = get_target_date()
    print(f"\n--- Creating Journal Entry for {vTargetDate} ---")
    
    # STEP 2: Collect Scores
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

    # STEP 3: Execute Workflow
    save_to_db(vTargetDate, vDailyScores)
    
    print("Generating Graph...")
    vImgFile = generate_radar_chart(vTargetDate, vDailyScores)
    
    print("Asking Gemini for analysis...")
    vAiText = get_ai_narrative(vTargetDate, vDailyScores)
    
    create_quarto_post(vTargetDate, vImgFile, vAiText)
    
    print(f"\nSUCCESS! Entry for {vTargetDate} created.")