import streamlit as st
import pandas as pd
import numpy as np
import os
import re
import ast
import joblib
import warnings
import plotly.express as px
import plotly.graph_objects as go
import streamlit.components.v1 as components
from collections import Counter

warnings.filterwarnings("ignore", category=UserWarning)

st.set_page_config(
    page_title="Job Market Intelligence",
    page_icon="assets/favicon.ico",
    layout="wide",
    initial_sidebar_state="collapsed"
)

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FIGURES_DIR = os.path.join(BASE_DIR, "figures")
MODELS_DIR = os.path.join(BASE_DIR, "models")
DATA_DIR = os.path.join(BASE_DIR, "data", "processed")

SKILL_LABELS = {
    "IT": "Information Technology", "SALE": "Sales", "MGMT": "Management",
    "ENG": "Engineering", "MNFC": "Manufacturing", "HCPR": "Healthcare & Practice",
    "FIN": "Finance", "BD": "Business Development", "OTHR": "Other",
    "ACCT": "Accounting", "ADM": "Administration", "MRKT": "Marketing",
    "PRJM": "Project Management", "ANLS": "Analytics", "LGL": "Legal",
    "HR": "Human Resources", "RSCH": "Research", "CUST": "Customer Service",
    "CNSL": "Consulting", "DSGN": "Design", "GENB": "General Business",
    "QA": "Quality Assurance", "PRDM": "Product Management", "EDU": "Education",
    "ART": "Arts & Creative", "TRNG": "Training", "STRA": "Strategy",
    "WRT": "Writing", "SUPL": "Supply Chain", "SCI": "Science"
}

EDU_LABELS = {0: "No Requirement", 1: "High School", 2: "Bachelor", 3: "Master", 4: "PhD"}
SEN_LABELS = {-1: "Unknown", 0: "Intern", 1: "Entry", 2: "Associate", 3: "Senior", 4: "Lead/Manager", 5: "Director+"}
SIZE_LABELS = {1: "1 to 10", 2: "11 to 50", 3: "51 to 200", 4: "201 to 500", 5: "501 to 1K", 6: "1K to 5K", 7: "5K plus"}

PALETTE = {
    "primary": "#00D4FF",
    "secondary": "#7C3AED",
    "accent": "#F59E0B",
    "success": "#10B981",
    "danger": "#EF4444",
    "bg": "#080B14",
    "card": "#0F1629",
    "border": "#1E2D4A",
    "text": "#E2E8F0",
    "muted": "#64748B"
}

NAV_ITEMS = {
    "Overview": "fa-chart-pie",
    "Market Data": "fa-database",
    "Skill Extractor": "fa-tags",
    "Job Galaxy": "fa-circle-nodes",
    "Salary Predictor": "fa-sliders"
}

PAGE_SLUGS = {
    "Overview": "overview",
    "Market Data": "market-data",
    "Skill Extractor": "skill-extractor",
    "Job Galaxy": "job-galaxy",
    "Salary Predictor": "salary-predictor"
}
SLUG_TO_PAGE = {v: k for k, v in PAGE_SLUGS.items()}

_CSS = """:root {
    --primary: #00D4FF;
    --secondary: #7C3AED;
    --accent: #F59E0B;
    --success: #10B981;
    --danger: #EF4444;
    --bg: #080B14;
    --card: #0F1629;
    --card2: #121D35;
    --border: #1E2D4A;
    --text: #E2E8F0;
    --muted: #64748B;
    --font-head: 'Plus Jakarta Sans', sans-serif;
    --font-body: 'Space Grotesk', sans-serif;
    --font-mono: 'DM Mono', monospace;
    --pad-x: 56px;
}
html, body, [class*="css"], .stApp {
    background-color: var(--bg) !important;
    color: var(--text) !important;
    font-family: var(--font-body) !important;
}
#root > div { background: var(--bg); }
.stApp { background: var(--bg) !important; }
header[data-testid="stHeader"] { display: none !important; }
section[data-testid="stSidebar"] { display: none !important; }
.block-container,
[data-testid="stMainBlockContainer"],
[data-testid="stAppViewBlockContainer"] {
    padding: 0 var(--pad-x) 80px !important;
    max-width: 100% !important;
    box-sizing: border-box !important;
    width: 100% !important;
}
button[data-testid="baseButton-headerNoPadding"] { display: none; }
div[data-testid="stToolbar"] { display: none; }
div[data-testid="stDecoration"] { display: none; }
div[data-testid="stStatusWidget"] { display: none; }
.viewerBadge_container__1QSob { display: none; }
footer { display: none !important; }
.nav-wrapper {
    position: sticky;
    top: 0;
    z-index: 9999;
    background: rgba(8, 11, 20, 0.92);
    backdrop-filter: blur(24px);
    -webkit-backdrop-filter: blur(24px);
    border-bottom: 1px solid var(--border);
    padding: 0 var(--pad-x);
    height: 64px;
    display: flex;
    align-items: center;
    justify-content: space-between;
    margin-left: calc(-1 * var(--pad-x));
    margin-right: calc(-1 * var(--pad-x));
    width: calc(100% + 2 * var(--pad-x));
    box-sizing: border-box;
}
.nav-logo {
    font-family: var(--font-head);
    font-size: 1.1rem;
    font-weight: 700;
    letter-spacing: 0.05em;
    color: var(--text);
    display: flex;
    align-items: center;
    gap: 10px;
}
.nav-logo span {
    background: linear-gradient(90deg, var(--primary), var(--secondary));
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}
.nav-dot {
    width: 8px; height: 8px;
    border-radius: 50%;
    background: var(--primary);
    box-shadow: 0 0 8px var(--primary);
    animation: pulse-dot 2s ease-in-out infinite;
}
@keyframes pulse-dot {
    0%, 100% { opacity: 1; transform: scale(1); }
    50% { opacity: 0.5; transform: scale(0.7); }
}
.nav-links { display: flex; gap: 4px; align-items: center; }
.nav-links a {
    font-family: var(--font-body);
    font-size: 0.83rem;
    font-weight: 500;
    text-decoration: none;
    border-radius: 8px;
    cursor: pointer;
    transition: all 0.2s;
    display: flex;
    align-items: center;
    gap: 7px;
    padding: 8px 14px;
    color: var(--muted);
}
.nav-links a:hover { color: var(--primary) !important; background: rgba(0,212,255,0.08); }
.nav-links a.active { color: var(--primary); background: rgba(0,212,255,0.08); }
.nav-team {
    font-size: 0.75rem;
    color: var(--muted);
    font-family: var(--font-mono);
}
.page-wrapper { padding-top: 32px; }
.hero-section {
    padding: 80px 0 60px;
    position: relative;
    overflow: hidden;
}
.hero-bg-grid {
    position: absolute;
    inset: 0;
    background-image: linear-gradient(rgba(0, 212, 255, 0.03) 1px, transparent 1px), linear-gradient(90deg, rgba(0, 212, 255, 0.03) 1px, transparent 1px);
    background-size: 48px 48px;
    pointer-events: none;
}
.hero-glow {
    position: absolute;
    top: -200px;
    left: 50%;
    transform: translateX(-50%);
    width: 800px;
    height: 600px;
    background: radial-gradient(ellipse, rgba(0, 212, 255, 0.08) 0%, rgba(124, 58, 237, 0.05) 40%, transparent 70%);
    pointer-events: none;
}
.hero-eyebrow {
    font-family: var(--font-mono);
    font-size: 0.75rem;
    letter-spacing: 0.2em;
    text-transform: uppercase;
    color: var(--primary);
    margin-bottom: 16px;
    display: flex;
    align-items: center;
    gap: 12px;
}
.hero-eyebrow::before {
    content: '';
    display: block;
    width: 32px;
    height: 1px;
    background: var(--primary);
}
.hero-title {
    font-family: var(--font-head);
    font-size: clamp(2.5rem, 5vw, 4.5rem);
    font-weight: 800;
    line-height: 1.05;
    margin: 0 0 24px;
    letter-spacing: -0.02em;
}
.hero-title .grad {
    background: linear-gradient(135deg, var(--primary) 0%, #6366F1 50%, var(--secondary) 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}
.hero-sub {
    font-size: 1.1rem;
    color: var(--muted);
    max-width: 560px;
    line-height: 1.7;
    font-weight: 300;
}
.kpi-row {
    display: grid;
    grid-template-columns: repeat(4, 1fr);
    gap: 16px;
    margin: 48px 0;
}
.kpi-card {
    background: var(--card);
    border: 1px solid var(--border);
    border-radius: 12px;
    padding: 24px;
    position: relative;
    overflow: hidden;
    transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
}
.kpi-card::before {
    content: '';
    position: absolute;
    inset: 0;
    background: linear-gradient(135deg, rgba(0, 212, 255, 0.05) 0%, transparent 60%);
    opacity: 0;
    transition: opacity 0.3s;
}
.kpi-card:hover { border-color: rgba(0, 212, 255, 0.4); transform: translateY(-2px); }
.kpi-card:hover::before { opacity: 1; }
.kpi-icon {
    width: 40px; height: 40px;
    border-radius: 10px;
    background: rgba(0, 212, 255, 0.1);
    display: flex; align-items: center; justify-content: center;
    margin-bottom: 16px;
    font-size: 1rem;
    color: var(--primary);
}
.kpi-value {
    font-family: var(--font-head);
    font-size: 2rem;
    font-weight: 700;
    color: var(--text);
    line-height: 1;
    margin-bottom: 4px;
}
.kpi-label {
    font-size: 0.75rem;
    color: var(--muted);
    text-transform: uppercase;
    letter-spacing: 0.1em;
    font-family: var(--font-mono);
}
.kpi-badge {
    position: absolute;
    top: 20px; right: 20px;
    font-size: 0.65rem;
    font-family: var(--font-mono);
    padding: 3px 8px;
    border-radius: 4px;
    background: rgba(16, 185, 129, 0.15);
    color: var(--success);
    border: 1px solid rgba(16, 185, 129, 0.3);
}
.section-header {
    display: flex;
    align-items: baseline;
    gap: 16px;
    margin: 48px 0 24px;
}
.section-title {
    font-family: var(--font-head);
    font-size: 1.5rem;
    font-weight: 700;
    color: var(--text);
}
.section-line {
    flex: 1;
    height: 1px;
    background: var(--border);
}
.section-count {
    font-family: var(--font-mono);
    font-size: 0.7rem;
    color: var(--muted);
}
.card {
    background: var(--card);
    border: 1px solid var(--border);
    border-radius: 16px;
    padding: 28px;
    transition: border-color 0.2s;
}
.card:hover { border-color: rgba(0, 212, 255, 0.2); }
.card-label {
    font-family: var(--font-mono);
    font-size: 0.68rem;
    letter-spacing: 0.15em;
    text-transform: uppercase;
    color: var(--muted);
    margin-bottom: 12px;
}
.card-title {
    font-family: var(--font-head);
    font-size: 1.1rem;
    font-weight: 600;
    color: var(--text);
    margin-bottom: 10px;
}
.card-body {
    font-size: 0.9rem;
    color: #94A3B8;
    line-height: 1.7;
}
.tag {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    padding: 4px 10px;
    border-radius: 6px;
    font-size: 0.75rem;
    font-family: var(--font-mono);
    background: rgba(0, 212, 255, 0.08);
    border: 1px solid rgba(0, 212, 255, 0.2);
    color: var(--primary);
    margin: 3px;
}
.tag-purple {
    background: rgba(124, 58, 237, 0.1);
    border-color: rgba(124, 58, 237, 0.3);
    color: #A78BFA;
}
.tag-amber {
    background: rgba(245, 158, 11, 0.1);
    border-color: rgba(245, 158, 11, 0.3);
    color: #FCD34D;
}
.tag-green {
    background: rgba(16, 185, 129, 0.1);
    border-color: rgba(16, 185, 129, 0.3);
    color: #34D399;
}
.pipeline-grid {
    display: grid;
    grid-template-columns: repeat(3, 1fr);
    gap: 20px;
    margin: 32px 0;
}
.pipeline-card {
    background: var(--card);
    border: 1px solid var(--border);
    border-radius: 16px;
    padding: 28px;
    position: relative;
    overflow: hidden;
    transition: all 0.3s;
}
.pipeline-card:hover { border-color: rgba(0, 212, 255, 0.3); transform: translateY(-3px); }
.pipeline-card::after {
    content: '';
    position: absolute;
    bottom: 0; left: 0; right: 0;
    height: 2px;
    opacity: 0;
    transition: opacity 0.3s;
}
.pipeline-card:hover::after { opacity: 1; }
.pc-a::after { background: linear-gradient(90deg, var(--primary), #6366F1); }
.pc-b::after { background: linear-gradient(90deg, var(--secondary), #EC4899); }
.pc-c::after { background: linear-gradient(90deg, var(--accent), var(--success)); }
.pipeline-num {
    font-family: var(--font-head);
    font-size: 3rem;
    font-weight: 800;
    opacity: 0.08;
    position: absolute;
    top: 16px; right: 20px;
    line-height: 1;
}
.pipeline-icon {
    width: 44px; height: 44px;
    border-radius: 12px;
    display: flex; align-items: center; justify-content: center;
    font-size: 1.1rem;
    margin-bottom: 16px;
}
.pi-a { background: rgba(0, 212, 255, 0.12); color: var(--primary); }
.pi-b { background: rgba(124, 58, 237, 0.12); color: #A78BFA; }
.pi-c { background: rgba(245, 158, 11, 0.12); color: var(--accent); }
.pipeline-label {
    font-family: var(--font-mono);
    font-size: 0.68rem;
    color: var(--muted);
    letter-spacing: 0.15em;
    text-transform: uppercase;
    margin-bottom: 6px;
}
.pipeline-title {
    font-family: var(--font-head);
    font-size: 1.1rem;
    font-weight: 600;
    color: var(--text);
    margin-bottom: 10px;
}
.pipeline-body {
    font-size: 0.87rem;
    color: #94A3B8;
    line-height: 1.65;
}
.team-grid {
    display: grid;
    grid-template-columns: repeat(4, 1fr);
    gap: 12px;
    margin-top: 20px;
}
.team-card {
    background: var(--card);
    border: 1px solid var(--border);
    border-radius: 12px;
    padding: 18px;
    text-align: center;
    transition: all 0.2s;
}
.team-card:hover { border-color: rgba(124, 58, 237, 0.4); }
.team-avatar {
    width: 48px; height: 48px;
    border-radius: 50%;
    background: linear-gradient(135deg, var(--secondary), var(--primary));
    display: flex; align-items: center; justify-content: center;
    font-family: var(--font-head);
    font-weight: 700;
    font-size: 1rem;
    color: white;
    margin: 0 auto 12px;
}
.team-name {
    font-size: 0.85rem;
    font-weight: 600;
    color: var(--text);
    margin-bottom: 4px;
}
.team-id {
    font-family: var(--font-mono);
    font-size: 0.65rem;
    color: var(--muted);
}
.info-row {
    display: flex;
    align-items: center;
    gap: 10px;
    padding: 12px 0;
    border-bottom: 1px solid rgba(30, 45, 74, 0.6);
    font-size: 0.88rem;
}
.info-row:last-child { border-bottom: none; }
.info-icon { color: var(--primary); width: 18px; text-align: center; }
.info-key { color: var(--muted); min-width: 140px; font-size: 0.8rem; }
.info-val { color: var(--text); }
.sim-result-box {
    background: var(--card);
    border: 1px solid var(--border);
    border-radius: 16px;
    padding: 36px;
    text-align: center;
    position: relative;
    overflow: hidden;
}
.sim-result-box::before {
    content: '';
    position: absolute;
    inset: 0;
    background: linear-gradient(135deg, rgba(0, 212, 255, 0.04) 0%, rgba(124, 58, 237, 0.03) 100%);
}
.sim-salary {
    font-family: var(--font-head);
    font-size: 3rem;
    font-weight: 800;
    line-height: 1;
    margin: 8px 0;
}
.sim-delta {
    font-size: 1rem;
    font-family: var(--font-mono);
}
.footer-bar {
    border-top: 1px solid var(--border);
    padding: 24px var(--pad-x);
    margin: 40px calc(-1 * var(--pad-x)) calc(-80px);
    display: flex;
    justify-content: space-between;
    align-items: center;
    background: var(--bg);
}
.stSelectbox > div > div {
    background: var(--card2) !important;
    border: 1px solid var(--border) !important;
    border-radius: 10px !important;
    color: var(--text) !important;
}
.stSlider > div { color: var(--text) !important; }
.stSlider [data-baseweb="slider"] { margin-top: 4px; }
div[data-testid="stSlider"] label { color: var(--muted) !important; font-size: 0.82rem !important; }
div[data-testid="stCheckbox"] label { color: var(--text) !important; }
.stButton > button {
    background: linear-gradient(135deg, var(--primary), #6366F1) !important;
    color: #080B14 !important;
    font-weight: 700 !important;
    font-family: var(--font-body) !important;
    border: none !important;
    border-radius: 10px !important;
    padding: 12px 28px !important;
    font-size: 0.9rem !important;
    letter-spacing: 0.04em !important;
    transition: all 0.2s !important;
    box-shadow: 0 0 20px rgba(0, 212, 255, 0.2) !important;
}
.stButton > button:hover {
    transform: translateY(-1px) !important;
    box-shadow: 0 0 32px rgba(0, 212, 255, 0.35) !important;
}
.stSelectbox label, .stMultiSelect label, .stTextInput label {
    color: var(--muted) !important;
    font-size: 0.8rem !important;
    text-transform: uppercase !important;
    letter-spacing: 0.1em !important;
}
div[data-testid="stImage"] img { border-radius: 12px; }
.stAlert { border-radius: 10px; }
div[data-testid="stMetric"] {
    background: var(--card);
    border: 1px solid var(--border);
    border-radius: 12px;
    padding: 16px;
}
[data-testid="stMetricValue"] {
    font-family: var(--font-head) !important;
    font-weight: 700 !important;
    color: var(--primary) !important;
}
[data-testid="stMetricLabel"] { color: var(--muted) !important; font-size: 0.78rem !important; }
div.stTabs [data-baseweb="tab-list"] {
    background: var(--card) !important;
    border-radius: 10px !important;
    padding: 4px !important;
    gap: 2px !important;
    border: 1px solid var(--border) !important;
}
div.stTabs [data-baseweb="tab"] {
    background: transparent !important;
    border-radius: 7px !important;
    color: var(--muted) !important;
    font-family: var(--font-body) !important;
    font-size: 0.85rem !important;
    font-weight: 500 !important;
    padding: 8px 16px !important;
}
div.stTabs [aria-selected="true"] {
    background: rgba(0, 212, 255, 0.12) !important;
    color: var(--primary) !important;
}
div.stTabs [data-baseweb="tab-highlight"] { background: transparent !important; }
div[data-testid="stExpander"] {
    background: var(--card) !important;
    border: 1px solid var(--border) !important;
    border-radius: 12px !important;
}
div[data-testid="stExpander"] summary {
    color: var(--text) !important;
    font-size: 0.9rem !important;
}"""

_clean_css = re.sub(r'\n{2,}', '\n', _CSS.strip())
st.markdown(
    '<link rel="preconnect" href="https://fonts.googleapis.com">'
    '<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>'
    '<link href="https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@300;400;500;600;700&family=DM+Mono:wght@400;500&family=Plus+Jakarta+Sans:wght@400;600;700;800&display=swap" rel="stylesheet">'
    '<link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.5.0/css/all.min.css">'
    f'<style>{_clean_css}</style>',
    unsafe_allow_html=True
)


def _sanitize_array_cols(df):
    for col in df.select_dtypes(include="object").columns:
        df[col] = df[col].apply(
            lambda x: str(list(x)) if isinstance(x, np.ndarray) else x
        )
    return df


@st.cache_data
def load_data():
    df = pd.read_parquet(os.path.join(DATA_DIR, "jobs_clean.parquet"))
    df = _sanitize_array_cols(df)
    return df

@st.cache_data
def load_image(filename):
    path = os.path.join(FIGURES_DIR, filename)
    return path if os.path.exists(path) else None

@st.cache_data
def compute_kpis(df):
    n_jobs = len(df)
    n_industries = df["industry"].nunique()
    median_sal = df["salary_mid"].median()
    remote_pct = (df["is_remote"].sum() / len(df)) * 100
    return n_jobs, n_industries, median_sal, remote_pct

@st.cache_data
def compute_skill_counts(df):
    all_skills = []
    for s in df["skills_list"].dropna():
        try:
            parsed = ast.literal_eval(s) if isinstance(s, str) else list(s)
            all_skills.extend(parsed)
        except Exception:
            pass
    return Counter(all_skills)

@st.cache_data
def industry_salary_stats(df):
    sal = df.dropna(subset=["salary_mid"])
    stats = sal.groupby("industry")["salary_mid"].agg(["mean", "median", "count"]).reset_index()
    stats.columns = ["industry", "mean_salary", "median_salary", "job_count"]
    return stats.sort_values("job_count", ascending=False)

@st.cache_data
def location_salary_stats(df):
    sal = df.dropna(subset=["salary_mid"])
    top_locs = sal["location"].value_counts().head(12).index
    stats = sal[sal["location"].isin(top_locs)].groupby("location")["salary_mid"].agg(["mean", "median", "count"]).reset_index()
    stats.columns = ["location", "mean_salary", "median_salary", "job_count"]
    return stats.sort_values("mean_salary", ascending=False)

@st.cache_data
def seniority_salary_stats(df):
    sal = df.dropna(subset=["salary_mid"])
    sal = sal[sal["seniority_encoded"] >= 0].copy()
    sal["seniority_label"] = sal["seniority_encoded"].map(SEN_LABELS)
    stats = sal.groupby(["seniority_encoded", "seniority_label"])["salary_mid"].agg(["mean", "median", "count"]).reset_index()
    stats.columns = ["seniority_encoded", "seniority_label", "mean_salary", "median_salary", "job_count"]
    return stats.sort_values("seniority_encoded")


def plotly_cfg(fig, height=340):
    fig.update_layout(
        height=height,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Space Grotesk, sans-serif", color="#94A3B8", size=11),
        margin=dict(l=16, r=16, t=32, b=16),
        xaxis=dict(showgrid=False, showline=False, zeroline=False, tickfont=dict(size=10)),
        yaxis=dict(showgrid=True, gridcolor="rgba(30,45,74,0.6)", showline=False, zeroline=False, tickfont=dict(size=10)),
        hoverlabel=dict(bgcolor="#0F1629", bordercolor="#1E2D4A", font=dict(family="Space Grotesk", size=12)),
        legend=dict(bgcolor="rgba(0,0,0,0)", bordercolor="rgba(0,0,0,0)")
    )
    return fig


try:
    df_global = load_data()
    DATA_LOADED = True
except Exception:
    DATA_LOADED = False
    df_global = None


slug = st.query_params.get("page", "overview")
if isinstance(slug, list):
    slug = slug[0] if slug else "overview"
page = SLUG_TO_PAGE.get(str(slug).lower(), "Overview")

nav_html = '<div class="nav-wrapper"><div class="nav-logo"><div class="nav-dot"></div><span>JOB</span> INTELLIGENCE</div><div class="nav-links">'
for item, icon in NAV_ITEMS.items():
    is_active = page == item
    active_cls = ' class="active"' if is_active else ''
    nav_html += f'<a href="?page={PAGE_SLUGS[item]}"{active_cls}><i class="fa-solid {icon}" style="font-size:0.8rem;"></i> {item}</a>'
nav_html += '</div><div class="nav-team">ML Project &bull; 2025/2026</div></div>'
st.markdown(nav_html, unsafe_allow_html=True)

st.markdown('<div class="page-wrapper">', unsafe_allow_html=True)


if page == "Overview":
    st.markdown("""
    <div class="hero-section">
        <div class="hero-bg-grid"></div>
        <div class="hero-glow"></div>
        <div class="hero-eyebrow">Machine Learning Final Project</div>
        <h1 class="hero-title">
            Job Market<br><span class="grad">Intelligence Platform</span>
        </h1>
        <p class="hero-sub">
            A multi-task NLP pipeline that extracts, clusters, and predicts insights from 10,000 real-world job postings using state-of-the-art deep learning and ensemble models.
        </p>
    </div>
    """, unsafe_allow_html=True)

    if DATA_LOADED:
        n_jobs, n_industries, median_sal, remote_pct = compute_kpis(df_global)
        st.markdown(f"""
        <div class="kpi-row">
            <div class="kpi-card">
                <div class="kpi-badge">Live Data</div>
                <div class="kpi-icon"><i class="fa-solid fa-briefcase"></i></div>
                <div class="kpi-value">{n_jobs:,}</div>
                <div class="kpi-label">Job Postings Analyzed</div>
            </div>
            <div class="kpi-card">
                <div class="kpi-icon" style="background:rgba(124,58,237,0.12); color:#A78BFA;"><i class="fa-solid fa-industry"></i></div>
                <div class="kpi-value">{n_industries}</div>
                <div class="kpi-label">Industries Covered</div>
            </div>
            <div class="kpi-card">
                <div class="kpi-icon" style="background:rgba(245,158,11,0.12); color:#F59E0B;"><i class="fa-solid fa-sack-dollar"></i></div>
                <div class="kpi-value">${median_sal/1000:.0f}K</div>
                <div class="kpi-label">Median Market Salary</div>
            </div>
            <div class="kpi-card">
                <div class="kpi-icon" style="background:rgba(16,185,129,0.12); color:#10B981;"><i class="fa-solid fa-wifi"></i></div>
                <div class="kpi-value">{remote_pct:.1f}%</div>
                <div class="kpi-label">Remote Positions</div>
            </div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("""
    <div class="section-header">
        <div class="section-title">NLP Pipeline Architecture</div>
        <div class="section-line"></div>
        <div class="section-count">3 Tasks</div>
    </div>
    <div class="pipeline-grid">
        <div class="pipeline-card pc-a">
            <div class="pipeline-num">A</div>
            <div class="pipeline-icon pi-a"><i class="fa-solid fa-tag"></i></div>
            <div class="pipeline-label">Task A</div>
            <div class="pipeline-title">Skill Extractor</div>
            <div class="pipeline-body">Fine-tuned JobBERT on 800 labeled sentences performs Named Entity Recognition to extract hard skills, software, and certifications from raw descriptions.</div>
        </div>
        <div class="pipeline-card pc-b">
            <div class="pipeline-num">B</div>
            <div class="pipeline-icon pi-b"><i class="fa-solid fa-circle-nodes"></i></div>
            <div class="pipeline-label">Task B</div>
            <div class="pipeline-title">Job Galaxy</div>
            <div class="pipeline-body">Sentence-BERT embeddings reduced via UMAP to 50D, then clustered using HDBSCAN to discover latent job role galaxies in the market landscape.</div>
        </div>
        <div class="pipeline-card pc-c">
            <div class="pipeline-num">C</div>
            <div class="pipeline-icon pi-c"><i class="fa-solid fa-calculator"></i></div>
            <div class="pipeline-label">Task C</div>
            <div class="pipeline-title">Salary Predictor</div>
            <div class="pipeline-body">Stacking ensemble of XGBoost and LightGBM combines NLP vectors, target-encoded categoricals, and engineered interaction features for salary regression.</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    if DATA_LOADED:
        st.markdown("""
        <div class="section-header">
            <div class="section-title">Market Snapshot</div>
            <div class="section-line"></div>
        </div>
        """, unsafe_allow_html=True)

        col1, col2 = st.columns(2)
        with col1:
            st.markdown('<div class="card"><div class="card-label">Top Roles by Volume</div>', unsafe_allow_html=True)
            top_titles = df_global["title"].value_counts().head(10).reset_index()
            top_titles.columns = ["title", "count"]
            fig = px.bar(
                top_titles, x="count", y="title", orientation="h",
                color="count",
                color_continuous_scale=["#1E2D4A", "#00D4FF"],
            )
            fig.update_traces(hovertemplate="<b>%{y}</b><br>Count: %{x}<extra></extra>")
            fig = plotly_cfg(fig, 320)
            fig.update_layout(yaxis=dict(autorange="reversed", categoryorder="total ascending"), coloraxis_showscale=False)
            st.plotly_chart(fig, width='stretch')
            st.markdown("</div>", unsafe_allow_html=True)

        with col2:
            st.markdown('<div class="card"><div class="card-label">Work Type Distribution</div>', unsafe_allow_html=True)
            wt = df_global["formatted_work_type"].value_counts().reset_index()
            wt.columns = ["type", "count"]
            fig2 = px.pie(
                wt, values="count", names="type",
                color_discrete_sequence=["#00D4FF", "#7C3AED", "#F59E0B", "#10B981", "#EF4444", "#6366F1"]
            )
            fig2.update_traces(
                textposition="outside", textinfo="percent+label",
                hovertemplate="<b>%{label}</b><br>%{value:,} jobs (%{percent})<extra></extra>",
                pull=[0.04] * len(wt)
            )
            fig2 = plotly_cfg(fig2, 320)
            st.plotly_chart(fig2, width='stretch')
            st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("""
    <div class="section-header">
        <div class="section-title">Development Team</div>
        <div class="section-line"></div>
    </div>
    <div class="team-grid">
        <div class="team-card">
            <div class="team-avatar">RF</div>
            <div class="team-name">Rizki Piji Fathoni</div>
            <div class="team-id">24031554029</div>
        </div>
        <div class="team-card">
            <div class="team-avatar">AJ</div>
            <div class="team-name">Alfin Jayadi</div>
            <div class="team-id">24031554082</div>
        </div>
        <div class="team-card">
            <div class="team-avatar">MR</div>
            <div class="team-name">Muhammad Rafi Fahrezi</div>
            <div class="team-id">24031554100</div>
        </div>
        <div class="team-card">
            <div class="team-avatar">DA</div>
            <div class="team-name">Daffa Ahmad Pangreksa</div>
            <div class="team-id">24031554159</div>
        </div>
    </div>
    """, unsafe_allow_html=True)


elif page == "Market Data":
    st.markdown("""
    <div class="hero-eyebrow" style="padding-top:24px;">Exploratory Data Analysis</div>
    <h2 style="font-family:var(--font-head); font-size:2.2rem; font-weight:800; letter-spacing:-0.02em; color:var(--text); margin-bottom:8px;">Market Intelligence</h2>
    <p style="color:var(--muted); font-size:0.95rem; margin-bottom:32px;">Interactive insights derived from 10,000 real job postings. All metrics are computed from actual dataset values.</p>
    """, unsafe_allow_html=True)

    if not DATA_LOADED:
        st.error("Dataset could not be loaded.")
    else:
        tab1, tab2, tab3, tab4 = st.tabs(["Salary Analysis", "Industry Breakdown", "Skills Profile", "EDA Figures"])

        with tab1:
            st.markdown('<div style="height:16px;"></div>', unsafe_allow_html=True)
            col_a, col_b = st.columns(2)

            with col_a:
                st.markdown('<div class="card"><div class="card-label">Salary Distribution</div>', unsafe_allow_html=True)
                sal_data = df_global.dropna(subset=["salary_mid"])
                sal_filtered = sal_data[sal_data["salary_mid"].between(30000, 300000)]
                fig = px.histogram(
                    sal_filtered, x="salary_mid", nbins=50,
                    color_discrete_sequence=["#00D4FF"]
                )
                fig.update_traces(
                    opacity=0.85,
                    hovertemplate="Salary Range: $%{x:,.0f}<br>Count: %{y}<extra></extra>"
                )
                fig.add_vline(x=sal_data["salary_mid"].median(), line_color="#F59E0B", line_width=2,
                              annotation_text=f"Median: ${sal_data['salary_mid'].median():,.0f}",
                              annotation_font_color="#F59E0B")
                fig = plotly_cfg(fig, 300)
                fig.update_layout(xaxis_title="Annual Salary (USD)", yaxis_title="Count")
                st.plotly_chart(fig, width='stretch')
                st.markdown("</div>", unsafe_allow_html=True)

            with col_b:
                st.markdown('<div class="card"><div class="card-label">Salary by Seniority Level</div>', unsafe_allow_html=True)
                sen_stats = seniority_salary_stats(df_global)
                fig2 = go.Figure()
                fig2.add_trace(go.Bar(
                    x=sen_stats["seniority_label"],
                    y=sen_stats["mean_salary"],
                    name="Mean",
                    marker_color="#00D4FF",
                    opacity=0.85,
                    hovertemplate="<b>%{x}</b><br>Mean: $%{y:,.0f}<extra></extra>"
                ))
                fig2.add_trace(go.Scatter(
                    x=sen_stats["seniority_label"],
                    y=sen_stats["median_salary"],
                    name="Median",
                    mode="lines+markers",
                    line=dict(color="#F59E0B", width=2),
                    marker=dict(size=7),
                    hovertemplate="<b>%{x}</b><br>Median: $%{y:,.0f}<extra></extra>"
                ))
                fig2 = plotly_cfg(fig2, 300)
                fig2.update_layout(yaxis_title="Salary (USD)", legend=dict(orientation="h", y=1.1))
                st.plotly_chart(fig2, width='stretch')
                st.markdown("</div>", unsafe_allow_html=True)

            st.markdown('<div class="card" style="margin-top:16px;"><div class="card-label">Salary by Location (Top 12 Markets)</div>', unsafe_allow_html=True)
            loc_stats = location_salary_stats(df_global)
            fig3 = go.Figure()
            fig3.add_trace(go.Bar(
                x=loc_stats["location"],
                y=loc_stats["mean_salary"],
                name="Mean Salary",
                marker=dict(
                    color=loc_stats["mean_salary"],
                    colorscale=[[0, "#1E2D4A"], [1, "#00D4FF"]],
                    showscale=False
                ),
                hovertemplate="<b>%{x}</b><br>Mean: $%{y:,.0f}<br>Jobs: %{customdata:,}<extra></extra>",
                customdata=loc_stats["job_count"]
            ))
            fig3.add_trace(go.Scatter(
                x=loc_stats["location"],
                y=loc_stats["median_salary"],
                name="Median Salary",
                mode="lines+markers",
                line=dict(color="#F59E0B", width=2),
                marker=dict(size=7),
                hovertemplate="<b>%{x}</b><br>Median: $%{y:,.0f}<extra></extra>"
            ))
            fig3 = plotly_cfg(fig3, 300)
            fig3.update_layout(xaxis_tickangle=-30, legend=dict(orientation="h", y=1.08))
            st.plotly_chart(fig3, width='stretch')
            st.markdown("</div>", unsafe_allow_html=True)

            st.markdown('<div style="height:16px;"></div>', unsafe_allow_html=True)
            col_c, col_d = st.columns(2)
            with col_c:
                st.markdown('<div class="card"><div class="card-label">Remote vs On-Site Salary Comparison</div>', unsafe_allow_html=True)
                sal_data["remote_label"] = sal_data["is_remote"].map({0: "On-Site", 1: "Remote"})
                fig_r = px.violin(
                    sal_data[sal_data["salary_mid"].between(30000, 250000)],
                    y="salary_mid", x="remote_label", color="remote_label",
                    box=True,
                    color_discrete_map={"On-Site": "#7C3AED", "Remote": "#00D4FF"}
                )
                fig_r.update_traces(hovertemplate="<b>%{x}</b><br>$%{y:,.0f}<extra></extra>")
                fig_r = plotly_cfg(fig_r, 300)
                fig_r.update_layout(showlegend=False, yaxis_title="Annual Salary (USD)")
                st.plotly_chart(fig_r, width='stretch')
                st.markdown("</div>", unsafe_allow_html=True)

            with col_d:
                st.markdown('<div class="card"><div class="card-label">Salary by Education Requirement</div>', unsafe_allow_html=True)
                edu_stats = sal_data.copy()
                edu_stats["edu_label"] = edu_stats["education_level"].map(EDU_LABELS)
                edu_agg = edu_stats.groupby("edu_label")["salary_mid"].median().reset_index()
                edu_order = ["No Requirement", "High School", "Bachelor", "Master", "PhD"]
                edu_agg["order"] = edu_agg["edu_label"].map({v: i for i, v in enumerate(edu_order)})
                edu_agg = edu_agg.sort_values("order")
                fig_e = px.bar(
                    edu_agg, x="edu_label", y="salary_mid",
                    color="salary_mid",
                    color_continuous_scale=["#1E2D4A", "#7C3AED", "#00D4FF"],
                )
                fig_e.update_traces(hovertemplate="<b>%{x}</b><br>Median: $%{y:,.0f}<extra></extra>")
                fig_e = plotly_cfg(fig_e, 300)
                fig_e.update_layout(yaxis_title="Median Salary", coloraxis_showscale=False, xaxis_title="")
                st.plotly_chart(fig_e, width='stretch')
                st.markdown("</div>", unsafe_allow_html=True)

        with tab2:
            st.markdown('<div style="height:16px;"></div>', unsafe_allow_html=True)
            ind_stats = industry_salary_stats(df_global)

            st.markdown('<div class="card"><div class="card-label">Top 20 Industries by Job Volume</div>', unsafe_allow_html=True)
            top20 = ind_stats.head(20)
            fig_ind = px.bar(
                top20, x="job_count", y="industry", orientation="h",
                color="median_salary",
                color_continuous_scale=[[0, "#1E2D4A"], [0.4, "#7C3AED"], [1, "#00D4FF"]],
                hover_data={"mean_salary": ":$,.0f", "median_salary": ":$,.0f", "job_count": ":,"}
            )
            fig_ind.update_traces(hovertemplate="<b>%{y}</b><br>Jobs: %{x:,}<br>Median Salary: $%{marker.color:,.0f}<extra></extra>")
            fig_ind = plotly_cfg(fig_ind, 420)
            fig_ind.update_layout(yaxis=dict(autorange="reversed"), coloraxis_colorbar=dict(title="Median Salary", tickformat="$,.0f"))
            st.plotly_chart(fig_ind, width='stretch')
            st.markdown("</div>", unsafe_allow_html=True)

            st.markdown('<div style="height:16px;"></div>', unsafe_allow_html=True)
            col_i1, col_i2 = st.columns(2)
            with col_i1:
                st.markdown('<div class="card"><div class="card-label">Top Industries by Median Salary</div>', unsafe_allow_html=True)
                top_sal_ind = ind_stats[ind_stats["job_count"] >= 30].nlargest(12, "median_salary")
                fig_si = px.bar(
                    top_sal_ind, x="median_salary", y="industry", orientation="h",
                    color_discrete_sequence=["#00D4FF"]
                )
                fig_si.update_traces(hovertemplate="<b>%{y}</b><br>Median: $%{x:,.0f}<extra></extra>")
                fig_si = plotly_cfg(fig_si, 360)
                fig_si.update_layout(yaxis=dict(autorange="reversed"), xaxis_title="Median Salary (USD)")
                st.plotly_chart(fig_si, width='stretch')
                st.markdown("</div>", unsafe_allow_html=True)

            with col_i2:
                st.markdown('<div class="card"><div class="card-label">Company Size Distribution</div>', unsafe_allow_html=True)
                df_global["size_label"] = df_global["company_size"].map(SIZE_LABELS)
                size_counts = df_global["size_label"].value_counts().reset_index()
                size_counts.columns = ["size", "count"]
                size_order = ["1 to 10", "11 to 50", "51 to 200", "201 to 500", "501 to 1K", "1K to 5K", "5K plus"]
                size_counts["order"] = size_counts["size"].map({v: i for i, v in enumerate(size_order)})
                size_counts = size_counts.sort_values("order")
                fig_sz = px.bar(
                    size_counts, x="size", y="count",
                    color="count",
                    color_continuous_scale=[[0, "#1E2D4A"], [1, "#7C3AED"]]
                )
                fig_sz.update_traces(hovertemplate="<b>%{x}</b><br>Jobs: %{y:,}<extra></extra>")
                fig_sz = plotly_cfg(fig_sz, 360)
                fig_sz.update_layout(xaxis_title="Company Size", coloraxis_showscale=False)
                st.plotly_chart(fig_sz, width='stretch')
                st.markdown("</div>", unsafe_allow_html=True)

        with tab3:
            st.markdown('<div style="height:16px;"></div>', unsafe_allow_html=True)
            skill_counts = compute_skill_counts(df_global)
            top_skills = skill_counts.most_common(25)
            skill_df = pd.DataFrame(top_skills, columns=["code", "count"])
            skill_df["label"] = skill_df["code"].map(SKILL_LABELS).fillna(skill_df["code"])

            st.markdown('<div class="card"><div class="card-label">Top 25 Skill Categories in Job Market</div>', unsafe_allow_html=True)
            fig_sk = px.bar(
                skill_df, x="count", y="label", orientation="h",
                color="count",
                color_continuous_scale=[[0, "#1E2D4A"], [0.5, "#6366F1"], [1, "#00D4FF"]]
            )
            fig_sk.update_traces(hovertemplate="<b>%{y}</b><br>Occurrences: %{x:,}<extra></extra>")
            fig_sk = plotly_cfg(fig_sk, 520)
            fig_sk.update_layout(yaxis=dict(autorange="reversed"), xaxis_title="Number of Job Postings", coloraxis_showscale=False)
            st.plotly_chart(fig_sk, width='stretch')
            st.markdown("</div>", unsafe_allow_html=True)

            st.markdown('<div style="height:16px;"></div>', unsafe_allow_html=True)
            st.markdown('<div class="card"><div class="card-label">Skill Demand vs Median Salary by Category</div>', unsafe_allow_html=True)
            df_skills_sal = df_global.dropna(subset=["salary_mid"]).copy()

            skill_sal_rows = []
            for code, label in SKILL_LABELS.items():
                mask = df_skills_sal["skills_list"].apply(
                    lambda x: code in (ast.literal_eval(x) if isinstance(x, str) and x else []) if pd.notna(x) else False
                )
                subset = df_skills_sal[mask]
                if len(subset) >= 20:
                    skill_sal_rows.append({
                        "code": code, "label": label,
                        "count": int(skill_counts.get(code, 0)),
                        "median_salary": float(subset["salary_mid"].median())
                    })

            if skill_sal_rows:
                skill_sal_df = pd.DataFrame(skill_sal_rows)
                fig_bubble = px.scatter(
                    skill_sal_df, x="count", y="median_salary",
                    size="count", text="label",
                    color="median_salary",
                    color_continuous_scale=[[0, "#7C3AED"], [0.5, "#6366F1"], [1, "#00D4FF"]],
                    size_max=45
                )
                fig_bubble.update_traces(
                    textposition="top center",
                    hovertemplate="<b>%{text}</b><br>Demand: %{x:,} jobs<br>Median Salary: $%{y:,.0f}<extra></extra>",
                    marker=dict(opacity=0.8, line=dict(width=1, color="rgba(255,255,255,0.1)"))
                )
                fig_bubble = plotly_cfg(fig_bubble, 400)
                fig_bubble.update_layout(
                    xaxis_title="Job Postings Requiring Skill",
                    yaxis_title="Median Annual Salary (USD)",
                    coloraxis_showscale=False
                )
                st.plotly_chart(fig_bubble, width='stretch')
            st.markdown("</div>", unsafe_allow_html=True)

            img_silver = load_image("eda_08_top_skills_silver.png")
            if img_silver:
                st.markdown('<div class="card" style="margin-top:16px;"><div class="card-label">Silver Standard NER Skill Extraction</div>', unsafe_allow_html=True)
                st.image(img_silver, width='stretch')
                st.markdown("</div>", unsafe_allow_html=True)

        with tab4:
            st.markdown('<div style="height:16px;"></div>', unsafe_allow_html=True)
            eda_figures = [
                ("eda_01_missing_values.png", "Missing Values Analysis"),
                ("eda_02_job_chars.png", "Job Posting Characteristics"),
                ("eda_03_wordcloud.png", "Description Word Cloud"),
                ("eda_04_top_titles.png", "Top Job Titles"),
                ("eda_05_salary_distribution.png", "Salary Distribution"),
                ("eda_06_desc_length.png", "Description Length Distribution"),
                ("eda_07_features.png", "Feature Engineering Overview"),
            ]
            pairs = [(eda_figures[i], eda_figures[i+1] if i+1 < len(eda_figures) else None) for i in range(0, len(eda_figures), 2)]
            for pair in pairs:
                col_f1, col_f2 = st.columns(2)
                with col_f1:
                    p = pair[0]
                    img = load_image(p[0])
                    if img:
                        st.markdown(f'<div class="card"><div class="card-label">{p[1]}</div>', unsafe_allow_html=True)
                        st.image(img, width='stretch')
                        st.markdown("</div>", unsafe_allow_html=True)
                with col_f2:
                    if pair[1]:
                        p = pair[1]
                        img = load_image(p[0])
                        if img:
                            st.markdown(f'<div class="card"><div class="card-label">{p[1]}</div>', unsafe_allow_html=True)
                            st.image(img, width='stretch')
                            st.markdown("</div>", unsafe_allow_html=True)


elif page == "Skill Extractor":
    st.markdown("""
    <div class="hero-eyebrow" style="padding-top:24px;">Task A</div>
    <h2 style="font-family:var(--font-head); font-size:2.2rem; font-weight:800; letter-spacing:-0.02em; color:var(--text); margin-bottom:8px;">Skill Extractor</h2>
    <p style="color:var(--muted); font-size:0.95rem; margin-bottom:32px;">Named Entity Recognition using fine-tuned JobBERT to extract hard skills, software knowledge, and certifications from unstructured job descriptions.</p>
    """, unsafe_allow_html=True)

    col_m1, col_m2, col_m3 = st.columns(3)
    with col_m1:
        st.markdown('<div class="card"><div class="card-label">Model</div><div class="card-title">JobBERT</div><div class="card-body">jjzha/jobbert-base-cased, fine-tuned on domain-specific job data.</div></div>', unsafe_allow_html=True)
    with col_m2:
        st.markdown('<div class="card"><div class="card-label">Training Data</div><div class="card-title">800 Sentences</div><div class="card-body">Manually annotated sentences with BIO-tagged entities for NER training.</div></div>', unsafe_allow_html=True)
    with col_m3:
        st.markdown('<div class="card"><div class="card-label">Pseudo-Labels</div><div class="card-title">10,000 Docs</div><div class="card-body">Model used to pseudo-label the full dataset, extracting entities at scale.</div></div>', unsafe_allow_html=True)

    st.markdown('<div style="height:24px;"></div>', unsafe_allow_html=True)

    img_ner = load_image("task_a_ner_top_entities.png")
    if img_ner:
        st.markdown('<div class="card"><div class="card-label">Top Extracted Entities by Type</div>', unsafe_allow_html=True)
        st.image(img_ner, width='stretch')
        st.markdown("</div>", unsafe_allow_html=True)

    st.markdown('<div style="height:16px;"></div>', unsafe_allow_html=True)

    if DATA_LOADED:
        st.markdown("""
        <div class="section-header">
            <div class="section-title">Live Skill Frequency Explorer</div>
            <div class="section-line"></div>
        </div>
        """, unsafe_allow_html=True)

        col_filter1, col_filter2 = st.columns([2, 1])
        with col_filter1:
            industry_options = ["All Industries"] + sorted(df_global["industry"].dropna().unique().tolist())
            selected_industry = st.selectbox("Filter by Industry", industry_options)
        with col_filter2:
            top_n = st.selectbox("Show Top N Skills", [10, 15, 20, 25], index=1)

        df_filtered = df_global if selected_industry == "All Industries" else df_global[df_global["industry"] == selected_industry]
        skill_counts_filtered = compute_skill_counts(df_filtered) if selected_industry == "All Industries" else Counter(
            [skill for skills in df_filtered["skills_list"].dropna()
             for skill in (ast.literal_eval(skills) if isinstance(skills, str) else list(skills))]
        )

        top_n_skills = skill_counts_filtered.most_common(top_n)
        sk_df = pd.DataFrame(top_n_skills, columns=["code", "count"])
        sk_df["label"] = sk_df["code"].map(SKILL_LABELS).fillna(sk_df["code"])
        sk_df["pct"] = (sk_df["count"] / len(df_filtered) * 100).round(1)

        st.markdown('<div class="card"><div class="card-label">Skill Demand Frequency</div>', unsafe_allow_html=True)
        fig_live = px.bar(
            sk_df, x="label", y="count",
            color="count",
            color_continuous_scale=[[0, "#1E2D4A"], [0.5, "#6366F1"], [1, "#00D4FF"]],
            custom_data=["pct"]
        )
        fig_live.update_traces(hovertemplate="<b>%{x}</b><br>Count: %{y:,}<br>Pct of Postings: %{customdata[0]:.1f}%<extra></extra>")
        fig_live = plotly_cfg(fig_live, 320)
        fig_live.update_layout(xaxis_tickangle=-30, coloraxis_showscale=False, xaxis_title="")
        st.plotly_chart(fig_live, width='stretch')
        st.markdown("</div>", unsafe_allow_html=True)

        st.markdown('<div style="height:16px;"></div>', unsafe_allow_html=True)

        with st.expander("View Skill Code Reference Table"):
            ref_df = pd.DataFrame([{"Code": k, "Full Name": v} for k, v in SKILL_LABELS.items()])
            ref_df["Count (All Data)"] = ref_df["Code"].map(lambda c: compute_skill_counts(df_global).get(c, 0))
            ref_df = ref_df.sort_values("Count (All Data)", ascending=False)
            st.dataframe(ref_df, width='stretch', hide_index=True)


elif page == "Job Galaxy":
    st.markdown("""
    <div class="hero-eyebrow" style="padding-top:24px;">Task B</div>
    <h2 style="font-family:var(--font-head); font-size:2.2rem; font-weight:800; letter-spacing:-0.02em; color:var(--text); margin-bottom:8px;">Job Galaxy</h2>
    <p style="color:var(--muted); font-size:0.95rem; margin-bottom:32px;">Unsupervised clustering of 10,000 job descriptions in latent semantic space using SBERT embeddings, UMAP dimensionality reduction, and HDBSCAN clustering.</p>
    """, unsafe_allow_html=True)

    col_m1, col_m2, col_m3 = st.columns(3)
    with col_m1:
        st.markdown('<div class="card"><div class="card-label">Embedding Model</div><div class="card-title">SBERT</div><div class="card-body">all-MiniLM-L6-v2 encodes full job descriptions into 384-dim semantic vectors.</div></div>', unsafe_allow_html=True)
    with col_m2:
        st.markdown('<div class="card"><div class="card-label">Dim Reduction</div><div class="card-title">UMAP 50D</div><div class="card-body">UMAP preserves local and global structure before clustering to avoid the curse of dimensionality.</div></div>', unsafe_allow_html=True)
    with col_m3:
        st.markdown('<div class="card"><div class="card-label">Clustering Algorithm</div><div class="card-title">HDBSCAN</div><div class="card-body">Finds arbitrary-shaped clusters and explicitly handles noise points as outliers.</div></div>', unsafe_allow_html=True)

    st.markdown("""
    <div class="section-header" style="margin-top:32px;">
        <div class="section-title">Interactive Job Map</div>
        <div class="section-line"></div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="card"><div class="card-label">HDBSCAN Clusters in 2D Semantic Space</div>', unsafe_allow_html=True)
    html_path = os.path.join(FIGURES_DIR, "cluster_04_hdbscan_plotly_scatter.html")
    if os.path.exists(html_path):
        with open(html_path, "r", encoding="utf-8") as f:
            html_data = f.read()
        components.html(html_data, height=580, scrolling=True)
    else:
        st.warning("Cluster map file not found.")
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown('<div style="height:16px;"></div>', unsafe_allow_html=True)

    cluster_figures = [
        ("cluster_01_hdbscan_sizes.png", "Cluster Size Distribution"),
        ("cluster_03_hdbscan_gmm_comparison.png", "HDBSCAN vs GMM Comparison"),
        ("cluster_05_gmm_confidence.png", "GMM Cluster Confidence"),
        ("cluster_06_skill_heatmap.png", "Skill Profile Heatmap per Cluster"),
        ("cluster_07_comparison_metrics.png", "Clustering Quality Metrics"),
        ("cluster_02_gmm_bic.png", "GMM BIC Curve"),
    ]
    pairs = [(cluster_figures[i], cluster_figures[i+1] if i+1 < len(cluster_figures) else None) for i in range(0, len(cluster_figures), 2)]
    for pair in pairs:
        col_c1, col_c2 = st.columns(2)
        with col_c1:
            p = pair[0]
            img = load_image(p[0])
            if img:
                st.markdown(f'<div class="card"><div class="card-label">{p[1]}</div>', unsafe_allow_html=True)
                st.image(img, width='stretch')
                st.markdown("</div>", unsafe_allow_html=True)
        with col_c2:
            if pair[1]:
                p = pair[1]
                img = load_image(p[0])
                if img:
                    st.markdown(f'<div class="card"><div class="card-label">{p[1]}</div>', unsafe_allow_html=True)
                    st.image(img, width='stretch')
                    st.markdown("</div>", unsafe_allow_html=True)


elif page == "Salary Predictor":
    st.markdown("""
    <div class="hero-eyebrow" style="padding-top:24px;">Task C</div>
    <h2 style="font-family:var(--font-head); font-size:2.2rem; font-weight:800; letter-spacing:-0.02em; color:var(--text); margin-bottom:8px;">Salary Intelligence</h2>
    <p style="color:var(--muted); font-size:0.95rem; margin-bottom:32px;">What-if salary simulation using a stacking ensemble (XGBoost + LightGBM) trained on NLP embeddings, encoded categoricals, and engineered interaction features.</p>
    """, unsafe_allow_html=True)

    col_m1, col_m2, col_m3, col_m4 = st.columns(4)
    with col_m1:
        st.markdown('<div class="card"><div class="card-label">Base Models</div><div class="card-title">XGBoost + LightGBM</div></div>', unsafe_allow_html=True)
    with col_m2:
        st.markdown('<div class="card"><div class="card-label">R2 Score</div><div class="card-title">0.65</div></div>', unsafe_allow_html=True)
    with col_m3:
        st.markdown('<div class="card"><div class="card-label">Feature Space</div><div class="card-title">SBERT + TF-IDF + Structured</div></div>', unsafe_allow_html=True)
    with col_m4:
        st.markdown('<div class="card"><div class="card-label">Salary Range</div><div class="card-title">$30K to $250K</div></div>', unsafe_allow_html=True)

    st.markdown("<div style='margin-bottom:20px;'></div>", unsafe_allow_html=True)
    
    tab_sim, tab_viz = st.tabs(["What-If Simulator", "Model Diagnostics"])

    with tab_viz:
        st.markdown('<div style="height:16px;"></div>', unsafe_allow_html=True)
        diag_figs = [
            ("task_c_feature_importance.png", "Feature Importance"),
            ("task_c_model_comparison.png", "Model Comparison"),
            ("task_c_stacking_diagnostics.png", "Stacking Diagnostics"),
        ]
        col_d1, col_d2 = st.columns(2)
        cols_diag = [col_d1, col_d2, col_d1]
        for i, (fname, label) in enumerate(diag_figs):
            img = load_image(fname)
            if img:
                with cols_diag[i]:
                    st.markdown(f'<div class="card"><div class="card-label">{label}</div>', unsafe_allow_html=True)
                    st.image(img, width='stretch')
                    st.markdown("</div>", unsafe_allow_html=True)

    with tab_sim:
        st.markdown('<div style="height:16px;"></div>', unsafe_allow_html=True)

        @st.cache_data
        def load_simulator_data():
            try:
                df = pd.read_parquet(os.path.join(DATA_DIR, "jobs_clean.parquet"))
                df = _sanitize_array_cols(df)
                df_reg = df.dropna(subset=["salary_mid"]).copy()
                df_reg = df_reg[df_reg["salary_mid"].between(30_000, 250_000)].reset_index(drop=True)

                sbert = np.load(os.path.join(DATA_DIR, "sbert_regression_embeddings.npy"))
                tfidf = np.load(os.path.join(DATA_DIR, "tfidf_svd_embeddings.npy"))

                te = joblib.load(os.path.join(MODELS_DIR, "target_encoder.joblib"))
                model = joblib.load(os.path.join(MODELS_DIR, "stacking.joblib"))

                def infer_seniority(title):
                    t = str(title).lower()
                    for pattern, level in [
                        (r"\b(intern|internship|co.?op)\b", 0),
                        (r"\b(entry.?level|junior|jr\.?|graduate|trainee)\b", 1),
                        (r"\b(associate|mid.?level|ii)\b", 2),
                        (r"\b(senior|sr\.?|staff|iii|iv)\b", 3),
                        (r"\b(lead|principal|manager|head|supervisor)\b", 4),
                        (r"\b(director|vp|vice|chief|cto|cfo|ceo|president)\b", 5),
                    ]:
                        if re.search(pattern, t):
                            return level
                    return -1

                df_reg["seniority_from_title"] = df_reg["title"].apply(infer_seniority)
                recovered = (df_reg["seniority_encoded"] == -1) & (df_reg["seniority_from_title"] != -1)
                df_reg.loc[recovered, "seniority_encoded"] = df_reg.loc[recovered, "seniority_from_title"]

                still_unknown = df_reg["seniority_encoded"] == -1
                ind_median = df_reg[~still_unknown].groupby("industry")["seniority_encoded"].median()
                glob_median = df_reg.loc[~still_unknown, "seniority_encoded"].median()
                df_reg["seniority_encoded"] = df_reg.apply(
                    lambda r: r["seniority_encoded"] if r["seniority_encoded"] != -1 else ind_median.get(r["industry"], glob_median),
                    axis=1
                ).round().astype(int)

                KEYWORDS = ["python", "sql", "aws", "azure", "gcp", "java", "javascript", "typescript", "react", "docker", "kubernetes", "spark", "machine learning", "deep learning", "data", "manager", "director", "lead", "senior", "principal", "architect", "agile", "scrum", "tableau", "excel", "r programming", "tensorflow", "pytorch", "compliance", "equity", "hedge fund", "private equity", "investment banking", "c++", "scala", "golang", "security", "devops", "mlops", "product manager", "data scientist", "data engineer", "software engineer", "quantitative", "physician", "dentist", "attorney", "surgeon", "pharmacist", "portfolio", "actuar", "underwriter", "cloud", "blockchain", "embedded"]
                kw_flags = np.array(df_reg["text_raw"].fillna("").str.lower().apply(lambda t: [1 if k in t else 0 for k in KEYWORDS]).tolist())
                kw_cols = [f"kw_{k.replace(' ', '_').replace('+', 'plus').replace('-', '_')}" for k in KEYWORDS]
                for i, c in enumerate(kw_cols):
                    df_reg[c] = kw_flags[:, i]

                df_reg["company_size_enc"] = pd.to_numeric(df_reg["company_size"], errors="coerce")
                df_reg["company_size_enc"] = df_reg["company_size_enc"].fillna(df_reg["company_size_enc"].median())
                for c in ["location", "industry"]:
                    if c in df_reg.columns:
                        df_reg[c] = df_reg[c].fillna("Unknown")

                df_reg_enc = df_reg.copy()
                df_reg_enc[[c + "_te" for c in ["location", "industry"]]] = te.transform(df_reg[["location", "industry"]])
                df_reg_enc = pd.get_dummies(df_reg_enc, columns=["formatted_work_type"], drop_first=True, dtype=int)
                work_type_cols = [c for c in df_reg_enc.columns if c.startswith("formatted_work_type_")]

                df_reg_enc["interact_sen_loc"] = df_reg_enc["seniority_encoded"] * df_reg_enc["location_te"]
                df_reg_enc["interact_sen_exp"] = df_reg_enc["seniority_encoded"] * df_reg_enc["years_exp_req"]
                df_reg_enc["interact_exp_ind"] = df_reg_enc["years_exp_req"] * df_reg_enc["industry_te"]

                num_cols = (
                    ["years_exp_req", "education_level", "seniority_encoded", "is_remote", "company_size_enc"]
                    + [c + "_te" for c in ["location", "industry"]]
                    + work_type_cols
                    + kw_cols
                    + ["interact_sen_loc", "interact_sen_exp", "interact_exp_ind"]
                )

                X_num = df_reg_enc[num_cols].fillna(0).values.astype(np.float32)
                X_all = np.hstack([sbert, tfidf, X_num])

                y = np.log1p(df_reg["salary_mid"].values)
                idx = np.arange(len(df_reg))
                salary_q = pd.qcut(df_reg["salary_mid"], q=5, labels=False)

                from sklearn.model_selection import train_test_split
                idx_t, idx_temp, _, y_temp = train_test_split(idx, y, test_size=0.3, random_state=42, stratify=salary_q)
                idx_val, idx_test, _, y_test = train_test_split(idx_temp, y_temp, test_size=0.5, random_state=42, stratify=salary_q.iloc[idx_temp])

                np.random.seed(42)
                sample_indices = np.random.choice(idx_test, 20, replace=False)
                sample_df = df_reg.iloc[sample_indices].copy()
                sample_X = X_all[sample_indices]

                return sample_df, sample_X, model, num_cols, sbert.shape[1], tfidf.shape[1]
            except Exception as e:
                st.error(f"Failed to load models or data: {e}")
                return None, None, None, None, None, None

        sample_df, sample_X, model, num_cols, sbert_dim, tfidf_dim = load_simulator_data()

        if sample_df is not None:
            col_sel, col_info = st.columns([3, 2])

            with col_sel:
                job_options = sample_df["title"].tolist()
                selected_title = st.selectbox("Select a Job Posting from Test Set", job_options)
                idx_in_sample = job_options.index(selected_title)
                row_data = sample_df.iloc[idx_in_sample]
                vector = sample_X[idx_in_sample].copy()
                true_salary = row_data["salary_mid"]
                original_pred = np.expm1(model.predict(vector.reshape(1, -1)))[0]

            with col_info:
                st.markdown(f"""
                <div class="card">
                    <div class="card-label">Job Details</div>
                    <div class="info-row">
                        <div class="info-icon"><i class="fa-solid fa-building"></i></div>
                        <div class="info-key">Industry</div>
                        <div class="info-val">{str(row_data["industry"])}</div>
                    </div>
                    <div class="info-row">
                        <div class="info-icon"><i class="fa-solid fa-location-dot"></i></div>
                        <div class="info-key">Location</div>
                        <div class="info-val">{str(row_data["location"])}</div>
                    </div>
                    <div class="info-row">
                        <div class="info-icon"><i class="fa-solid fa-wifi"></i></div>
                        <div class="info-key">Remote</div>
                        <div class="info-val">{"Yes" if row_data["is_remote"] else "No"}</div>
                    </div>
                    <div class="info-row">
                        <div class="info-icon"><i class="fa-solid fa-graduation-cap"></i></div>
                        <div class="info-key">Education</div>
                        <div class="info-val">{EDU_LABELS.get(int(row_data["education_level"]), "Unknown")}</div>
                    </div>
                </div>
                """, unsafe_allow_html=True)

            st.markdown('<div style="height:16px;"></div>', unsafe_allow_html=True)

            col_t, col_p = st.columns(2)
            with col_t:
                st.markdown(f"""
                <div class="card" style="text-align:center;">
                    <div class="card-label">Actual Market Salary</div>
                    <div style="font-family:var(--font-head); font-size:2.2rem; font-weight:800; color:#10B981;">${true_salary:,.0f}</div>
                    <div style="font-size:0.8rem; color:var(--muted); margin-top:4px;">From job dataset</div>
                </div>
                """, unsafe_allow_html=True)
            with col_p:
                st.markdown(f"""
                <div class="card" style="text-align:center;">
                    <div class="card-label">Model Base Prediction</div>
                    <div style="font-family:var(--font-head); font-size:2.2rem; font-weight:800; color:var(--primary);">${original_pred:,.0f}</div>
                    <div style="font-size:0.8rem; color:var(--muted); margin-top:4px;">Stacking ensemble output</div>
                </div>
                """, unsafe_allow_html=True)

            st.markdown("""
            <div class="section-header" style="margin-top:28px;">
                <div class="section-title">Adjust Feature Values</div>
                <div class="section-line"></div>
            </div>
            """, unsafe_allow_html=True)

            col_tw1, col_tw2 = st.columns(2)
            with col_tw1:
                new_exp = st.slider(
                    "Years of Experience Required",
                    0, 15,
                    int(row_data["years_exp_req"] if not pd.isna(row_data["years_exp_req"]) else 0)
                )
                new_remote = st.checkbox("Remote Position", bool(row_data["is_remote"]))
            with col_tw2:
                new_sen = st.slider(
                    "Seniority Level  (0 Intern to 5 Executive)",
                    0, 5,
                    int(row_data["seniority_encoded"])
                )
                new_edu = st.slider(
                    "Education Level  (0 None to 4 PhD)",
                    0, 4,
                    int(row_data["education_level"] if not pd.isna(row_data["education_level"]) else 0)
                )

            st.markdown('<div style="height:8px;"></div>', unsafe_allow_html=True)

            if st.button("Run Salary Simulation", width='stretch'):
                num_idx_exp = sbert_dim + tfidf_dim + num_cols.index("years_exp_req")
                num_idx_rem = sbert_dim + tfidf_dim + num_cols.index("is_remote")
                num_idx_edu = sbert_dim + tfidf_dim + num_cols.index("education_level")

                vector[num_idx_exp] = new_exp
                vector[num_idx_rem] = 1 if new_remote else 0
                vector[num_idx_edu] = new_edu

                for lvl in range(1, 6):
                    col_name = f"seniority_encoded_{lvl}"
                    if col_name in num_cols:
                        vector[sbert_dim + tfidf_dim + num_cols.index(col_name)] = 1 if new_sen == lvl else 0

                num_idx_int_exp = sbert_dim + tfidf_dim + num_cols.index("interact_sen_exp")
                num_idx_int_loc = sbert_dim + tfidf_dim + num_cols.index("interact_sen_loc")

                loc_te = row_data["location_te"] if "location_te" in row_data else vector[sbert_dim + tfidf_dim + num_cols.index("location_te")]

                vector[num_idx_int_exp] = new_sen * new_exp
                vector[num_idx_int_loc] = new_sen * loc_te

                new_pred = np.expm1(model.predict(vector.reshape(1, -1)))[0]
                diff = new_pred - original_pred
                color = "#10B981" if diff >= 0 else "#EF4444"
                sign = "+" if diff >= 0 else ""
                arrow = "fa-arrow-up" if diff >= 0 else "fa-arrow-down"

                st.markdown(f"""
                <div class="sim-result-box">
                    <div class="card-label">Simulated Prediction</div>
                    <div class="sim-salary" style="color:{color};">${new_pred:,.0f}</div>
                    <div class="sim-delta" style="color:{color};">
                        <i class="fa-solid {arrow}"></i>
                        {sign}${diff:,.0f} from base prediction
                    </div>
                </div>
                """, unsafe_allow_html=True)

                fig_comp = go.Figure(go.Bar(
                    x=["Actual Salary", "Base Prediction", "Simulated Prediction"],
                    y=[true_salary, original_pred, new_pred],
                    marker=dict(
                        color=["#10B981", "#00D4FF", color],
                        opacity=0.85
                    ),
                    hovertemplate="<b>%{x}</b><br>$%{y:,.0f}<extra></extra>"
                ))
                fig_comp = plotly_cfg(fig_comp, 260)
                fig_comp.update_layout(yaxis_title="Annual Salary (USD)", showlegend=False)
                st.plotly_chart(fig_comp, width='stretch')

st.markdown("</div>", unsafe_allow_html=True)

st.markdown("""
<div class="footer-bar">
    <div style="font-family:var(--font-head); font-weight:700; color:var(--text);">
        <span style="background: linear-gradient(90deg, var(--primary), var(--secondary)); -webkit-background-clip: text; -webkit-text-fill-color: transparent;">JOB</span> INTELLIGENCE
    </div>
    <div style="font-family:var(--font-mono); font-size:0.72rem; color:var(--muted);">
        Machine Learning Final Project &bull; Even Semester 2025/2026 &bull; 029 082 100 159
    </div>
    <div style="font-family:var(--font-mono); font-size:0.72rem; color:var(--muted);">
        10,000 Job Postings &bull; 3 NLP Pipelines &bull; All Data Real
    </div>
</div>
""", unsafe_allow_html=True)