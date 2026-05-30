import streamlit as st
import pandas as pd
import numpy as np
import os
import joblib

# --- PAGE CONFIG ---
st.set_page_config(
    page_title="Job Market Intelligence",
    page_icon="💼",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- CUSTOM CSS ---
st.markdown("""
<style>
    body { color: #E2E8F0; background-color: #0F172A; }
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700&display=swap');
    * { font-family: 'Inter', sans-serif; }
    h1, h2, h3 {
        background: -webkit-linear-gradient(45deg, #38BDF8, #818CF8);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-weight: 700 !important;
    }
    .glass-card {
        background: rgba(30, 41, 59, 0.7); backdrop-filter: blur(10px); -webkit-backdrop-filter: blur(10px);
        border: 1px solid rgba(255, 255, 255, 0.1); border-radius: 16px; padding: 24px; margin-bottom: 24px;
        box-shadow: 0 4px 30px rgba(0, 0, 0, 0.1); transition: transform 0.3s ease;
    }
    .glass-card:hover { transform: translateY(-5px); border: 1px solid rgba(56, 189, 248, 0.4); }
    .metric-value { font-size: 2.5rem; font-weight: 700; color: #38BDF8; text-shadow: 0 0 10px rgba(56,189,248,0.3); }
    .metric-label { font-size: 1rem; color: #94A3B8; text-transform: uppercase; letter-spacing: 1px; }
    [data-testid="stSidebar"] { background-color: #0B1120; border-right: 1px solid #1E293B; }
    .team-badge { display: inline-block; padding: 6px 12px; background: linear-gradient(90deg, #1E293B, #0F172A); border-left: 3px solid #818CF8; border-radius: 4px; margin: 4px; font-size: 0.9rem; color: #CBD5E1; }
</style>
""", unsafe_allow_html=True)

# --- PATHS ---
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FIGURES_DIR = os.path.join(BASE_DIR, 'figures')
MODELS_DIR = os.path.join(BASE_DIR, 'models')
DATA_DIR = os.path.join(BASE_DIR, 'data', 'processed')

st.sidebar.title("Job Intelligence")
st.sidebar.markdown("---")

page = st.sidebar.radio(
    "Navigation", 
    ["Project Overview", "Skill Extractor", "Job Galaxy (Clustering)", "Salary Predictor"]
)

st.sidebar.markdown("---")
st.sidebar.markdown("### Development Team")
st.sidebar.markdown("""
- Rizki Piji Fathoni (24031554029)
- Alfin Jayadi (24031554082)
- Muhammad Rafi Fahrezi (24031554100)
- Daffa Ahmad Pangreksa (24031554159)
""")

@st.cache_data
def load_image(filename):
    path = os.path.join(FIGURES_DIR, filename)
    return path if os.path.exists(path) else None

if page == "Project Overview":
    st.markdown("<h1>Job Market Intelligence: Multi-Task NLP Pipeline</h1>", unsafe_allow_html=True)
    st.markdown("<p style='font-size: 1.2rem; color: #94A3B8;'>Machine Learning Final Project | Even Semester 2025/2026</p>", unsafe_allow_html=True)
    
    st.markdown("""
    <div class='glass-card'>
        <h3>Project Vision & Impact 🚀</h3>
        <p>In the modern job market, raw job descriptions contain massive amounts of unstructured insights that are difficult to analyze at scale. 
        This project tackles that challenge by building a <b>Multi-Task Natural Language Processing (NLP) Pipeline</b>. 
        Instead of relying on simple keyword matching, we utilize state-of-the-art Deep Learning models to truly understand the context of job postings.</p>
        
        <h4>How it Works:</h4>
        
- **Task A (Skill Extractor):** We use a fine-tuned JobBERT model to automatically read job descriptions and extract the required Hard Skills, Software Knowledge, and Certifications.
- **Task B (Job Galaxy):** We convert textual descriptions into mathematical vectors using Sentence-BERT, then cluster them using HDBSCAN to discover latent "Galaxies" of similar job roles in the market.
- **Task C (Salary Simulator):** We combine the NLP vectors with structured data (like Industry and Experience) into an XGBoost & LightGBM ensemble to predict the fair market salary of a job posting.
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns(3)
    with col1: st.markdown("<div class='glass-card'><div class='metric-value'>10K+</div><div class='metric-label'>Job Postings</div></div>", unsafe_allow_html=True)
    with col2: st.markdown("<div class='glass-card'><div class='metric-value'>3</div><div class='metric-label'>NLP Pipelines</div></div>", unsafe_allow_html=True)
    with col3: st.markdown("<div class='glass-card'><div class='metric-value'>0.65</div><div class='metric-label'>R² Salary Score</div></div>", unsafe_allow_html=True)
        
    st.markdown("---")
    st.markdown("### 📊 Market Exploratory Data Analysis")
    
    col_a, col_b = st.columns(2)
    with col_a:
        img_roles = load_image('eda_04_top_titles.png')
        if img_roles: st.image(img_roles, caption="Top Job Roles in the Dataset", width='stretch')
    with col_b:
        img_sal = load_image('eda_05_salary_distribution.png')
        if img_sal: st.image(img_sal, caption="Salary Distribution across Market", width='stretch')

elif page == "Skill Extractor":
    st.markdown("<h1>Task A: Skill, Knowledge, and Certification Extraction</h1>", unsafe_allow_html=True)
    
    st.markdown("""
    <div class='glass-card'>
        <h3>Methodology 🛠️</h3>
        <p>We fine-tuned <b>JobBERT (jjzha/jobbert-base-cased)</b> on a manually labeled dataset (800 sentences) to perform Named Entity Recognition (NER). 
        The model was then used to pseudo-label the remaining 10,000 job descriptions, successfully extracting thousands of hard skills, software knowledge, and certifications from unstructured text.</p>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("### 🏆 Top Extracted Entities")
    img_ner = load_image('task_a_ner_top_entities.png')
    if img_ner: st.image(img_ner, width='stretch')
        
    st.markdown("### 🥈 Silver Standard Verification")
    img_silver = load_image('eda_08_top_skills_silver.png')
    if img_silver: st.image(img_silver, width='stretch')

elif page == "Job Galaxy (Clustering)":
    st.markdown("<h1>Task B: Job Galaxy (Unsupervised Clustering)</h1>", unsafe_allow_html=True)
    
    st.markdown("""
    <div class='glass-card'>
        <h3>How We Grouped Jobs 🧩</h3>
        <p>We converted job descriptions into dense vectors using <b>Sentence-BERT (all-MiniLM-L6-v2)</b>. 
        Because clustering in high dimensions suffers from the curse of dimensionality, we reduced the vectors to 50D using <b>UMAP</b>. 
        Finally, we applied <b>HDBSCAN</b> to discover arbitrary-shaped clusters and filter out noise.</p>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("### 🔭 Interactive Job Map (HDBSCAN)")
    html_path = os.path.join(FIGURES_DIR, 'cluster_04_hdbscan_plotly_scatter.html')
    if os.path.exists(html_path):
        with open(html_path, 'r', encoding='utf-8') as f:
            html_data = f.read()
        import streamlit.components.v1 as components
        components.html(html_data, height=600, scrolling=True)
    else:
        st.warning("Plotly map not found in figures directory.")
        
    st.markdown("### 🔥 Skill Profile per Cluster (NER Fusion)")
    st.markdown("This heatmap combines the extracted skills from Task A with the clusters from Task B, revealing the technical DNA of each job galaxy.")
    img_clus = load_image('cluster_01_hdbscan_sizes.png')
    if img_clus: st.image(img_clus, width='stretch')

elif page == "Salary Predictor":
    st.markdown("<h1>Task C: Salary Intelligence Simulator</h1>", unsafe_allow_html=True)
    
    st.markdown("""
    <div class='glass-card'>
        <h3>Predictive Engine ⚙️</h3>
        <p>This engine uses a Stacking Ensemble (XGBoost + LightGBM) trained on numerical features, target-encoded categoricals, TF-IDF SVD, and 1000-character SBERT embeddings. 
        It predicts the median salary for a job posting. Explore the feature importance below!</p>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("### 🔍 Model Feature Importance")
    img_feat = load_image('task_c_feature_importance.png')
    if img_feat: st.image(img_feat, width='stretch')
        
    st.markdown("---")
    st.title("What-If Analysis Simulator")
    st.markdown("Select a job posting from the unseen Test Set, then tweak the numerical variables to instantly simulate the salary change!")
    
    @st.cache_data
    def load_simulator_data():
        try:
            df = pd.read_parquet(os.path.join(DATA_DIR, 'jobs_clean.parquet'))
            df_reg = df.dropna(subset=["salary_mid"]).copy()
            df_reg = df_reg[df_reg["salary_mid"].between(30_000, 250_000)].reset_index(drop=True)
            
            sbert = np.load(os.path.join(DATA_DIR, 'sbert_regression_embeddings.npy'))
            tfidf = np.load(os.path.join(DATA_DIR, 'tfidf_svd_embeddings.npy'))
            
            te = joblib.load(os.path.join(MODELS_DIR, 'target_encoder.joblib'))
            model = joblib.load(os.path.join(MODELS_DIR, 'stacking.joblib'))
            
            def infer_seniority(title):
                import re
                t = str(title).lower()
                for pattern, level in [
                    (r'\b(intern|internship|co.?op)\b', 0),
                    (r'\b(entry.?level|junior|jr\.?|graduate|trainee)\b', 1),
                    (r'\b(associate|mid.?level|ii)\b', 2),
                    (r'\b(senior|sr\.?|staff|iii|iv)\b', 3),
                    (r'\b(lead|principal|manager|head|supervisor)\b', 4),
                    (r'\b(director|vp|vice|chief|cto|cfo|ceo|president)\b', 5),
                ]:
                    if re.search(pattern, t): return level
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
            for i, c in enumerate(kw_cols): df_reg[c] = kw_flags[:, i]
            
            df_reg["company_size_enc"] = pd.to_numeric(df_reg["company_size"], errors="coerce")
            df_reg["company_size_enc"] = df_reg["company_size_enc"].fillna(df_reg["company_size_enc"].median())
            for c in ["location", "industry"]:
                if c in df_reg.columns: df_reg[c] = df_reg[c].fillna("Unknown")
            
            df_reg_enc = df_reg.copy()
            df_reg_enc[[c + "_te" for c in ["location", "industry"]]] = te.transform(df_reg[["location", "industry"]])
            df_reg_enc = pd.get_dummies(df_reg_enc, columns=["formatted_work_type"], drop_first=True, dtype=int)
            work_type_cols = [c for c in df_reg_enc.columns if c.startswith("formatted_work_type_")]
            
            # Interaction MUST be calculated before One-Hot Encoding seniority!
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
            
            # Target
            y = np.log1p(df_reg["salary_mid"].values)
            idx = np.arange(len(df_reg))
            salary_q = pd.qcut(df_reg["salary_mid"], q=5, labels=False)
            
            from sklearn.model_selection import train_test_split
            idx_t, idx_temp, _, y_temp = train_test_split(idx, y, test_size=0.3, random_state=42, stratify=salary_q)
            idx_val, idx_test, _, y_test = train_test_split(idx_temp, y_temp, test_size=0.5, random_state=42, stratify=salary_q.iloc[idx_temp])
            
            # Prepare a sample of 20 random jobs from test set for the simulator
            np.random.seed(42)
            sample_indices = np.random.choice(idx_test, 20, replace=False)
            sample_df = df_reg.iloc[sample_indices].copy()
            sample_X = X_all[sample_indices]
            
            return sample_df, sample_X, model, num_cols, sbert.shape[1], tfidf.shape[1]
        except Exception as e:
            st.error(f"Failed to load models/data: {e}")
            return None, None, None, None, None, None

    sample_df, sample_X, model, num_cols, sbert_dim, tfidf_dim = load_simulator_data()
    
    if sample_df is not None:
        job_options = sample_df['title'].tolist()
        selected_title = st.selectbox("1. Select Job Posting to Simulate:", job_options)
        
        idx_in_sample = job_options.index(selected_title)
        row_data = sample_df.iloc[idx_in_sample]
        vector = sample_X[idx_in_sample].copy()
        
        true_salary = row_data['salary_mid']
        original_pred = np.expm1(model.predict(vector.reshape(1, -1)))[0]
        
        st.markdown(f"**Industry:** {row_data['industry']} | **Location:** {row_data['location']}")
        st.markdown(f"**Original Text Snippet:** _{str(row_data['text_raw'])[:150]}..._")
        
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("<div class='glass-card'><h4 style='margin:0'>True Salary</h4><h2 style='color:#A7F3D0'>${:,.0f}</h2></div>".format(true_salary), unsafe_allow_html=True)
        with col2:
            st.markdown("<div class='glass-card'><h4 style='margin:0'>Base Prediction</h4><h2 style='color:#38BDF8'>${:,.0f}</h2></div>".format(original_pred), unsafe_allow_html=True)
            
        st.markdown("### 2. Tweak Features")
        tc1, tc2 = st.columns(2)
        with tc1:
            new_exp = st.slider("Years of Experience Required", 0, 15, int(row_data['years_exp_req'] if not pd.isna(row_data['years_exp_req']) else 0))
            new_remote = st.checkbox("Is Remote?", bool(row_data['is_remote']))
        with tc2:
            new_sen = st.slider("Seniority Level (0=Intern, 5=Executive)", 0, 5, int(row_data['seniority_encoded']))
            new_edu = st.slider("Education Level (0=None, 2=Masters)", 0, 2, int(row_data['education_level'] if not pd.isna(row_data['education_level']) else 0))
            
        if st.button("Run Simulation", use_container_width=True):
            # Update the numerical vector
            num_idx_exp = sbert_dim + tfidf_dim + num_cols.index("years_exp_req")
            num_idx_rem = sbert_dim + tfidf_dim + num_cols.index("is_remote")
            num_idx_edu = sbert_dim + tfidf_dim + num_cols.index("education_level")
            
            vector[num_idx_exp] = new_exp
            vector[num_idx_rem] = 1 if new_remote else 0
            vector[num_idx_edu] = new_edu
            
            # Update One-Hot Encoded Seniority (columns: seniority_encoded_1 to 5)
            for lvl in range(1, 6):
                col_name = f"seniority_encoded_{lvl}"
                if col_name in num_cols:
                    vector[sbert_dim + tfidf_dim + num_cols.index(col_name)] = 1 if new_sen == lvl else 0
            
            # Update Interactions (recalculated based on base te_cols and new values)
            num_idx_int_exp = sbert_dim + tfidf_dim + num_cols.index("interact_sen_exp")
            num_idx_int_loc = sbert_dim + tfidf_dim + num_cols.index("interact_sen_loc")
            
            # Retrieve the pre-computed location_te for this row
            loc_te = row_data['location_te'] if 'location_te' in row_data else vector[sbert_dim + tfidf_dim + num_cols.index('location_te')]
            
            vector[num_idx_int_exp] = new_sen * new_exp
            vector[num_idx_int_loc] = new_sen * loc_te
            
            new_pred = np.expm1(model.predict(vector.reshape(1, -1)))[0]
            
            diff = new_pred - original_pred
            color = "#A7F3D0" if diff > 0 else "#FDA4AF"
            sign = "+" if diff > 0 else ""
            
            st.markdown(f"<div class='glass-card' style='text-align:center'><h3>Simulated Prediction</h3><h1 style='color:{color}'>${new_pred:,.0f} <span style='font-size:1.2rem'>({sign}${diff:,.0f})</span></h1></div>", unsafe_allow_html=True)
