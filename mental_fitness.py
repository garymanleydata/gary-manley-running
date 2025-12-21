import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import io
import textwrap

# --- PAGE CONFIG ---
st.set_page_config(page_title="Mental Performance Hub", layout="wide", page_icon="🧠")

# --- CSS FOR CARDS ---
st.markdown("""
<style>
    .resource-card {
        background-color: #f0f2f6;
        padding: 20px;
        border-radius: 10px;
        margin-bottom: 10px;
        border: 1px solid #e0e0e0;
    }
    .tag {
        display: inline-block;
        background-color: #000;
        color: #fff;
        padding: 2px 8px;
        border-radius: 4px;
        font-size: 0.8em;
        margin-right: 5px;
    }
</style>
""", unsafe_allow_html=True)

st.title("🧠 Mental Performance Hub")
st.markdown("Knowledge is confidence. Curate your mental toolkit from elite resources.")

# --- DATA: THE RESOURCE LIBRARY ---
# In a real app, this could be loaded from a CSV/JSON file
resources = [
    {
        "Title": "The 100% Rule",
        "Author": "Eliud Kipchoge",
        "Type": "Video",
        "Category": "Discipline",
        "Link": "https://www.youtube.com/watch?v=2JqJ7tVqjyw", # Placeholder/Real link
        "Desc": "Why 100% disciplined is easier than 99%."
    },
    {
        "Title": "How to Master Self-Talk",
        "Author": "Dr. Nate Zinsser",
        "Type": "Book Summary",
        "Category": "Confidence",
        "Link": "https://www.youtube.com/watch?v=example",
        "Desc": "Summary of Chapter 4: The Doorway technique."
    },
    {
        "Title": "Embracing the Pain Cave",
        "Author": "Courtney Dauwalter",
        "Type": "Interview",
        "Category": "Pain Tolerance",
        "Link": "https://www.youtube.com/watch?v=example",
        "Desc": "Visualizing pain as a physical place you enter to work."
    },
    {
        "Title": "The Central Governor Theory",
        "Author": "Tim Noakes",
        "Type": "Article",
        "Category": "Science",
        "Link": "https://en.wikipedia.org/wiki/Central_governor",
        "Desc": "Understanding why your brain slows you down before your body fails."
    },
    {
        "Title": "Visualizing the Perfect Race",
        "Author": "Marius Bakken",
        "Type": "Technique",
        "Category": "Race Prep",
        "Link": "https://www.mariusbakken.com/",
        "Desc": "The Norwegian approach to pre-race visualization."
    }
]

df_res = pd.DataFrame(resources)

# Initialize Session State
if "saved_insights" not in st.session_state:
    st.session_state["saved_insights"] = [
        "Pain is just information.",
        "I control the controllables."
    ]

# --- SIDEBAR NAVIGATION ---
mode = st.sidebar.radio("Navigation:", ["📚 Resource Library", "🛠️ My Dashboard Builder"])

# ==============================================================================
# MODULE 1: THE RESOURCE LIBRARY (NETFLIX STYLE)
# ==============================================================================
if mode == "📚 Resource Library":
    
    # 1. SEARCH & FILTER
    c1, c2 = st.columns([3, 1])
    with c1:
        search = st.text_input("🔍 Search Library", placeholder="e.g. pain, anxiety, kipchoge...")
    with c2:
        cat_filter = st.selectbox("Filter by Topic", ["All"] + list(df_res['Category'].unique()))

    # 2. FILTER LOGIC
    df_show = df_res.copy()
    if cat_filter != "All":
        df_show = df_show[df_show['Category'] == cat_filter]
    if search:
        df_show = df_show[df_show['Title'].str.contains(search, case=False) | df_show['Desc'].str.contains(search, case=False)]

    st.divider()

    # 3. DISPLAY GRID
    # We use columns to create a grid layout
    cols = st.columns(2) # 2 Cards per row
    
    for idx, row in df_show.iterrows():
        col = cols[idx % 2]
        with col:
            with st.container(border=True):
                # Tag & Title
                st.markdown(f"**{row['Title']}**")
                st.caption(f"{row['Author']} • {row['Type']}")
                
                # Content (Video or Link)
                if row['Type'] == "Video" or row['Type'] == "Interview":
                    # Display video player if it's a video
                    st.video(row['Link'])
                else:
                    # Link button for articles
                    st.link_button(f"Read {row['Type']}", row['Link'])
                
                st.markdown(f"_{row['Desc']}_")
                
                # "Save Insight" Interaction
                with st.expander("✍️ Take Notes / Save Insight"):
                    note = st.text_input("What did you learn?", key=f"note_{idx}")
                    if st.button("Save to Dashboard", key=f"save_{idx}"):
                        if note:
                            st.session_state['saved_insights'].append(note)
                            st.toast("Insight Saved!")

# ==============================================================================
# MODULE 2: DASHBOARD BUILDER (MANUAL + SAVED)
# ==============================================================================
elif mode == "🛠️ My Dashboard Builder":
    st.header("🛠️ Construct Your Mental Board")
    st.markdown("Combine your saved insights into a printable manifesto.")
    
    c1, c2 = st.columns(2)
    
    with c1:
        st.subheader("Your Saved Insights")
        if st.session_state['saved_insights']:
            for i, insight in enumerate(st.session_state['saved_insights']):
                col_txt, col_del = st.columns([0.9, 0.1])
                col_txt.success(f"• {insight}")
                if col_del.button("X", key=f"del_ins_{i}"):
                    st.session_state['saved_insights'].pop(i)
                    st.rerun()
        else:
            st.info("Go to the Library tab to watch videos and save insights!")
            
    with c2:
        st.subheader("Add Custom Affirmation")
        custom = st.text_input("Manual Entry")
        if st.button("Add"):
            if custom:
                st.session_state['saved_insights'].append(custom)
                st.rerun()

    st.divider()
    
    # VISUAL GENERATOR (Using the fixed code from previous step)
    if st.button("Generate Printable Board"):
        # Setup Figure
        fig, ax = plt.subplots(figsize=(16, 11))
        ax.set_xlim(0, 100)
        ax.set_ylim(0, 100)
        ax.axis('off')
        
        # Geometry
        Y_TOP = 20; H_TOP = 70
        
        # Draw Logic
        def draw_box(x, y, w, h, color, title, lines):
            shadow = patches.Rectangle((x+0.5, y-0.5), w, h, linewidth=0, facecolor=(0,0,0,0.2))
            ax.add_patch(shadow)
            rect = patches.Rectangle((x, y), w, h, linewidth=2, edgecolor='black', facecolor=color)
            ax.add_patch(rect)
            ax.text(x + w/2, y + h - 3, title.upper(), ha='center', va='top', fontsize=18, fontweight='bold', fontfamily='sans-serif')
            
            chars = int(w * 1.8)
            y_cursor = y + h - 10 
            for line in lines:
                wrapped = textwrap.wrap(f"• {line}", width=chars)
                for wl in wrapped:
                    if y_cursor > y + 2: 
                        ax.text(x + 2, y_cursor, wl, ha='left', va='top', fontsize=14, fontfamily='sans-serif')
                        y_cursor -= 3

        # Header
        ax.text(50, 95, "MY MENTAL PERFORMANCE BOARD", ha='center', va='center', fontsize=32, fontweight='bold')
        
        # 1. The Big Box (Insights)
        draw_box(10, 10, 80, 80, '#fff9c4', "KEY STRATEGIES & AFFIRMATIONS", st.session_state['saved_insights'])

        buf = io.BytesIO()
        plt.savefig(buf, format="png", bbox_inches='tight', dpi=250, facecolor='#f0f2f6')
        buf.seek(0)
        
        st.image(buf, caption="Preview", use_column_width=True)
        st.download_button("Download PNG", data=buf, file_name="mental_board.png", mime="image/png")