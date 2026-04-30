"""
CareerPath Optimizer
======================
A Greedy and Dynamic Programming Based Learning Roadmap Generator.

This Streamlit application helps students select an optimized learning
roadmap by applying two classic algorithms — Greedy (value/hour ratio)
and 0/1 Knapsack Dynamic Programming — to a curated course dataset.
"""

import streamlit as st
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # Set backend before importing pyplot
import matplotlib.pyplot as plt
import copy

from utils.data_loader import load_courses, filter_courses
from algorithms.greedy_selector import greedy_select_courses
from algorithms.dp_knapsack import dp_optimize_courses
from algorithms.cpp_wrapper import is_cpp_available, greedy_optimize_cpp, dp_optimize_cpp

# ──────────────────────────────────────────────
# Page Configuration
# ──────────────────────────────────────────────
st.set_page_config(
    page_title="CareerPath Optimizer",
    page_icon=None,
    layout="wide",
    initial_sidebar_state="expanded",
)

# ──────────────────────────────────────────────
# Custom CSS for a polished, professional look
# ──────────────────────────────────────────────
st.markdown("""
<style>
    /* ---------- Google Font ---------- */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

    /* ---------- Root variables ---------- */
    :root {
        --primary: #6C63FF;
        --primary-dark: #5A52D5;
        --accent: #00D2FF;
        --success: #00C896;
        --warning: #FFB347;
        --bg-dark: #0E1117;
        --card-bg: rgba(255, 255, 255, 0.03);
        --card-border: rgba(255, 255, 255, 0.1);
        --text-primary: #EAEAEA;
        --text-muted: #9CA3AF;
    }

    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }

    /* ---------- Hero header ---------- */
    .hero-title {
        font-size: 2.8rem;
        font-weight: 800;
        background: linear-gradient(135deg, #6C63FF 0%, #00D2FF 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        margin-bottom: 0;
        letter-spacing: -0.5px;
    }
    .hero-subtitle {
        font-size: 1.15rem;
        color: var(--text-muted);
        text-align: center;
        margin-top: 4px;
        margin-bottom: 28px;
        font-weight: 400;
    }

    /* ---------- Glass card ---------- */
    .glass-card {
        background: var(--card-bg);
        border: 1px solid var(--card-border);
        border-radius: 12px;
        padding: 24px;
        margin-bottom: 20px;
        backdrop-filter: blur(10px);
        transition: border-color 0.3s ease;
    }
    .glass-card:hover {
        border-color: var(--primary);
    }

    /* ---------- Section header ---------- */
    .section-header {
        font-size: 1.5rem;
        font-weight: 700;
        color: var(--text-primary);
        margin-bottom: 12px;
        display: flex;
        align-items: center;
        gap: 10px;
    }

    /* ---------- Metric pill ---------- */
    .metric-row {
        display: flex;
        gap: 16px;
        flex-wrap: wrap;
        margin: 14px 0;
    }
    .metric-pill {
        background: linear-gradient(135deg, rgba(108,99,255,0.15), rgba(0,210,255,0.10));
        border: 1px solid rgba(108,99,255,0.3);
        border-radius: 12px;
        padding: 14px 22px;
        text-align: center;
        min-width: 140px;
        flex: 1;
    }
    .metric-pill .metric-value {
        font-size: 1.8rem;
        font-weight: 700;
        color: var(--accent);
    }
    .metric-pill .metric-label {
        font-size: 0.82rem;
        color: var(--text-muted);
        text-transform: uppercase;
        letter-spacing: 0.5px;
        margin-top: 2px;
    }

    /* ---------- Roadmap step ---------- */
    .roadmap-step {
        display: flex;
        align-items: flex-start;
        gap: 14px;
        margin-bottom: 14px;
    }
    .step-number {
        background: linear-gradient(135deg, #6C63FF, #00D2FF);
        color: #fff;
        font-weight: 700;
        min-width: 36px;
        height: 36px;
        border-radius: 50%;
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 0.95rem;
        flex-shrink: 0;
    }
    .step-content {
        flex: 1;
    }
    .step-title {
        font-weight: 600;
        color: var(--text-primary);
        font-size: 1rem;
    }
    .step-meta {
        font-size: 0.82rem;
        color: var(--text-muted);
        margin-top: 2px;
    }

    /* ---------- Algorithm explanation card ---------- */
    .algo-card {
        background: linear-gradient(135deg, rgba(108,99,255,0.08), rgba(0,210,255,0.05));
        border: 1px solid rgba(108,99,255,0.2);
        border-radius: 14px;
        padding: 22px 26px;
        margin-bottom: 16px;
    }
    .algo-card h4 {
        color: var(--accent);
        margin-bottom: 8px;
    }
    .algo-card p, .algo-card li {
        color: var(--text-muted);
        font-size: 0.92rem;
        line-height: 1.6;
    }

    /* ---------- Comparison badge ---------- */
    .badge-greedy {
        background: rgba(255, 179, 71, 0.15);
        color: #FFB347;
        padding: 4px 12px;
        border-radius: 20px;
        font-size: 0.78rem;
        font-weight: 600;
    }
    .badge-dp {
        background: rgba(0, 200, 150, 0.15);
        color: #00C896;
        padding: 4px 12px;
        border-radius: 20px;
        font-size: 0.78rem;
        font-weight: 600;
    }

    /* ---------- Sidebar styling ---------- */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #12141D 0%, #1A1D2E 100%);
    }
    [data-testid="stSidebar"] .stSelectbox label,
    [data-testid="stSidebar"] .stSlider label {
        color: var(--text-primary) !important;
        font-weight: 600;
    }

    /* ---------- Divider ---------- */
    .custom-divider {
        height: 1px;
        background: linear-gradient(90deg, transparent, rgba(108,99,255,0.4), transparent);
        margin: 30px 0;
    }

    /* ---------- Preference chip ---------- */
    .pref-chip {
        display: inline-block;
        background: rgba(108,99,255,0.12);
        border: 1px solid rgba(108,99,255,0.3);
        color: var(--text-primary);
        padding: 6px 16px;
        border-radius: 20px;
        font-size: 0.88rem;
        font-weight: 500;
        margin: 4px 4px;
    }
</style>
""", unsafe_allow_html=True)


# ──────────────────────────────────────────────
# Header
# ──────────────────────────────────────────────
st.markdown('<h1 class="hero-title">CareerPath Optimizer</h1>', unsafe_allow_html=True)
st.markdown('<p class="hero-subtitle">Algorithm-Based Learning Roadmap Generator</p>', unsafe_allow_html=True)

# Engine Status Badge
if is_cpp_available():
    st.markdown(
        """
        <div style="text-align:center; margin-bottom: 20px;">
            <span style="background: rgba(0, 210, 255, 0.1); color: #00D2FF; border: 1px solid #00D2FF; padding: 4px 12px; border-radius: 20px; font-size: 0.75rem; font-weight: 700; letter-spacing: 1px; text-transform: uppercase;">
                🚀 C++ Engine Active
            </span>
        </div>
        """,
        unsafe_allow_html=True
    )
else:
    st.markdown(
        """
        <div style="text-align:center; margin-bottom: 20px;">
            <span style="background: rgba(255, 179, 71, 0.1); color: #FFB347; border: 1px solid #FFB347; padding: 4px 12px; border-radius: 20px; font-size: 0.75rem; font-weight: 700; letter-spacing: 1px; text-transform: uppercase;">
                ⚠️ Python Engine (C++ Unavailable)
            </span>
        </div>
        """,
        unsafe_allow_html=True
    )

# ──────────────────────────────────────────────
# Sidebar — User Inputs
# ──────────────────────────────────────────────
with st.sidebar:
    st.markdown("## Configuration")
    st.markdown("---")

    career_goal = st.selectbox(
        "Career Goal",
        options=[
            "Full Stack Developer",
            "Data Scientist",
            "AI/ML Engineer",
            "DevOps Engineer",
        ],
        index=0,
    )

    skill_level = st.selectbox(
        "Skill Level",
        options=["Beginner", "Intermediate", "Advanced"],
        index=0,
    )

    max_hours = st.slider(
        "Available Study Hours",
        min_value=5,
        max_value=100,
        value=30,
        step=1,
    )

    st.markdown("---")
    st.markdown(
        "<small style='color:#9CA3AF;'>Powered by <b>Greedy</b> & <b>DP Knapsack</b> algorithms</small>",
        unsafe_allow_html=True,
    )

# ──────────────────────────────────────────────
# 1. Project Overview
# ──────────────────────────────────────────────
st.markdown('<div class="glass-card">', unsafe_allow_html=True)
st.markdown('<div class="section-header">Project Overview</div>', unsafe_allow_html=True)
st.markdown("""
This application helps students build an **optimized learning roadmap** tailored to their
career goals, skill level, and available study time. It applies two classic algorithms:

- **Greedy Algorithm** — Picks courses with the highest *value-per-hour* ratio first for a
  quick, efficient recommendation.
- **Dynamic Programming (0/1 Knapsack)** — Finds the mathematically *optimal* combination
  of courses that maximizes total value within the time budget.

Compare both results side-by-side and choose the plan that works best for you!
""")
st.markdown('</div>', unsafe_allow_html=True)

# ──────────────────────────────────────────────
# 2. Selected User Preferences
# ──────────────────────────────────────────────
st.markdown('<div class="custom-divider"></div>', unsafe_allow_html=True)
st.markdown('<div class="section-header">Your Preferences</div>', unsafe_allow_html=True)

st.markdown(
    f"""
    <div style="display:flex;flex-wrap:wrap;gap:6px;">
        <span class="pref-chip">{career_goal}</span>
        <span class="pref-chip">{skill_level}</span>
        <span class="pref-chip">{max_hours} hours</span>
    </div>
    """,
    unsafe_allow_html=True,
)

# ──────────────────────────────────────────────
# Load & Filter Courses
# ──────────────────────────────────────────────
all_courses = load_courses()
filtered = filter_courses(all_courses, career_goal, skill_level)

# ──────────────────────────────────────────────
# 3. Available Courses
# ──────────────────────────────────────────────
st.markdown('<div class="custom-divider"></div>', unsafe_allow_html=True)
st.markdown('<div class="section-header">Available Courses</div>', unsafe_allow_html=True)

if not filtered:
    st.warning("No courses found for the selected career goal and skill level.")
    st.stop()

courses_df = pd.DataFrame(filtered)[["id", "title", "hours", "value", "description"]]
courses_df.index = range(1, len(courses_df) + 1)
st.dataframe(courses_df, width="stretch")

# ──────────────────────────────────────────────
# Run Algorithms (deep copies to avoid mutation)
# ──────────────────────────────────────────────
# --- Greedy selection ---
if is_cpp_available():
    greedy_courses, greedy_hours, greedy_value = greedy_optimize_cpp(
        copy.deepcopy(filtered), max_hours
    )
else:
    greedy_courses, greedy_hours, greedy_value = greedy_select_courses(
        copy.deepcopy(filtered), max_hours
    )

# --- DP selection ---
if is_cpp_available():
    dp_courses, dp_hours, dp_value, dp_table = dp_optimize_cpp(
        copy.deepcopy(filtered), max_hours
    )
else:
    dp_courses, dp_hours, dp_value, dp_table = dp_optimize_courses(
        copy.deepcopy(filtered), max_hours
    )

# ──────────────────────────────────────────────
# 4 & 5. Side-by-Side Comparison
# ──────────────────────────────────────────────
st.markdown('<div class="custom-divider"></div>', unsafe_allow_html=True)
st.markdown('<div class="section-header">Algorithm Results</div>', unsafe_allow_html=True)

col_greedy, col_dp = st.columns(2)

# --- Helper to render a result column ---
def render_result(col, title, badge_class, courses, total_hours, total_value):
    with col:
        st.markdown(
            f'<div class="glass-card">'
            f'<div class="section-header">{title} <span class="{badge_class}">'
            f'{"Fast" if "greedy" in badge_class else "Optimal"}</span></div>',
            unsafe_allow_html=True,
        )

        # Metrics
        st.markdown(
            f"""
            <div class="metric-row">
                <div class="metric-pill">
                    <div class="metric-value">{total_hours}</div>
                    <div class="metric-label">Total Hours</div>
                </div>
                <div class="metric-pill">
                    <div class="metric-value">{total_value}</div>
                    <div class="metric-label">Total Value</div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        if courses:
            # Table of selected courses
            result_data = []
            for c in courses:
                ratio = round(c["value"] / c["hours"], 2)
                result_data.append({
                    "Title": c["title"],
                    "Hours": c["hours"],
                    "Value": c["value"],
                    "Val/Hr": ratio,
                    "Description": c["description"],
                })
            df = pd.DataFrame(result_data)
            df.index = range(1, len(df) + 1)
            st.dataframe(df, width="stretch")
        else:
            st.info("No courses could be selected within the given hours.")

        st.markdown("</div>", unsafe_allow_html=True)


render_result(col_greedy, "Greedy Algorithm", "badge-greedy",
              greedy_courses, greedy_hours, greedy_value)
render_result(col_dp, "DP Knapsack", "badge-dp",
              dp_courses, dp_hours, dp_value)

# ──────────────────────────────────────────────
# 6. Learning Roadmap (from DP result)
# ──────────────────────────────────────────────
st.markdown('<div class="custom-divider"></div>', unsafe_allow_html=True)
st.markdown('<div class="section-header">Optimized Learning Roadmap</div>', unsafe_allow_html=True)

roadmap_courses = dp_courses if dp_courses else greedy_courses

if roadmap_courses:
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    for idx, course in enumerate(roadmap_courses, start=1):
        st.markdown(
            f"""
            <div class="roadmap-step">
                <div class="step-number">{idx}</div>
                <div class="step-content">
                    <div class="step-title">{course['title']}</div>
                    <div class="step-meta">{course['hours']} hrs &nbsp;·&nbsp; Value: {course['value']} &nbsp;·&nbsp; {course['description']}</div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    st.markdown("</div>", unsafe_allow_html=True)
else:
    st.info("Increase available study hours to generate a roadmap.")

# ──────────────────────────────────────────────
# 7. Comparison Table
# ──────────────────────────────────────────────
st.markdown('<div class="custom-divider"></div>', unsafe_allow_html=True)
st.markdown('<div class="section-header">Comparison: Greedy vs DP</div>', unsafe_allow_html=True)

comparison_df = pd.DataFrame({
    "Algorithm": ["Greedy", "Dynamic Programming"],
    "Total Hours": [greedy_hours, dp_hours],
    "Total Value": [greedy_value, dp_value],
    "Nature": [
        "Fast but not always optimal",
        "Optimal but uses more computation",
    ],
})
comparison_df.index = range(1, len(comparison_df) + 1)

st.markdown('<div class="glass-card">', unsafe_allow_html=True)
st.table(comparison_df)
st.markdown('</div>', unsafe_allow_html=True)

# ── Simple bar chart comparing the two algorithms ──
fig, axes = plt.subplots(1, 2, figsize=(8, 3), dpi=100)
fig.patch.set_facecolor("#0E1117")

colors = ["#FFB347", "#00C896"]
labels = ["Greedy", "DP"]

for ax, metric, values, ylabel in zip(
    axes,
    ["Total Hours Used", "Total Value Achieved"],
    [[greedy_hours, dp_hours], [greedy_value, dp_value]],
    ["Hours", "Value"],
):
    bars = ax.bar(labels, values, color=colors, edgecolor="none", width=0.5)
    ax.set_title(metric, color="#EAEAEA", fontsize=11, fontweight=600)
    ax.set_ylabel(ylabel, color="#9CA3AF", fontsize=9)
    ax.set_facecolor("#0E1117")
    ax.tick_params(colors="#9CA3AF")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#2A2D3E")
    ax.spines["bottom"].set_color("#2A2D3E")
    for bar, val in zip(bars, values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.3,
            str(val),
            ha="center",
            color="#EAEAEA",
            fontweight=600,
            fontsize=10,
        )

fig.tight_layout()
st.pyplot(fig)
plt.close(fig)  # Free memory immediately

# ──────────────────────────────────────────────
# 8. Algorithm Explanations
# ──────────────────────────────────────────────
st.markdown('<div class="custom-divider"></div>', unsafe_allow_html=True)
st.markdown('<div class="section-header">How the Algorithms Work</div>', unsafe_allow_html=True)

col_e1, col_e2 = st.columns(2)

with col_e1:
    st.markdown("""
    <div class="algo-card">
        <h4>Greedy Algorithm</h4>
        <p><b>Strategy:</b> Always pick the course with the best value-per-hour ratio next.</p>
        <ol>
            <li>Calculate <code>ratio = value / hours</code> for each course.</li>
            <li>Sort courses by ratio in descending order.</li>
            <li>Select courses one by one while total hours ≤ budget.</li>
        </ol>
        <p><b>Time Complexity:</b> O(n log n) — dominated by sorting.</p>
        <p><b>Trade-off:</b> Very fast, but may skip a slightly worse-ratio course that
        could unlock a better overall combination.</p>
    </div>
    """, unsafe_allow_html=True)

with col_e2:
    st.markdown("""
    <div class="algo-card">
        <h4>0/1 Knapsack — Dynamic Programming</h4>
        <p><b>Strategy:</b> Evaluate every possible include/exclude combination using a
        DP table to guarantee the maximum total value.</p>
        <ol>
            <li>Build a table <code>dp[i][w]</code> — best value using first <em>i</em> courses with capacity <em>w</em>.</li>
            <li>For each course, decide: <em>take it</em> or <em>skip it</em>.</li>
            <li>Backtrack from <code>dp[n][W]</code> to recover selected courses.</li>
        </ol>
        <p><b>Time Complexity:</b> O(n × W) where W = available hours.</p>
        <p><b>Trade-off:</b> Always optimal, but requires more memory and computation
        for large inputs.</p>
    </div>
    """, unsafe_allow_html=True)

# ──────────────────────────────────────────────
# Footer
# ──────────────────────────────────────────────
st.markdown('<div class="custom-divider"></div>', unsafe_allow_html=True)
st.markdown(
    """
    <div style="text-align:center;color:#9CA3AF;font-size:0.82rem;padding-bottom:20px;">
        <b>CareerPath Optimizer</b> · Built with Streamlit · Greedy & DP Algorithms<br>
        This project demonstrates the practical use of Greedy Algorithm and Dynamic Programming
        for solving a real-world learning path optimization problem.
    </div>
    """,
    unsafe_allow_html=True,
)
