# Job Market Intelligence: Multi-Task NLP Pipeline

This repository contains the final project for the Machine Learning course (Even Semester 2025/2026). It implements a comprehensive Multi-Task Natural Language Processing (NLP) pipeline designed to extract insights, cluster roles, and predict salaries from raw job postings.

🚀 **[Live Dashboard Deployment](https://rizki0907-job-market-intelligence---ml-proj-dashboardapp-rbjn6g.streamlit.app/)**

## Project Vision & Impact

In the modern job market, raw job descriptions contain massive amounts of unstructured insights that are difficult to analyze at scale. This project tackles that challenge by building a Multi-Task NLP Pipeline. Instead of relying on simple keyword matching, we utilize state-of-the-art Deep Learning models to truly understand the context of job postings.

## Dataset

The data driving this project originates from the **[LinkedIn Job Postings Dataset (2023 - 2024)](https://www.kaggle.com/datasets/arshkon/linkedin-job-postings/data)** available on Kaggle. This comprehensive dataset contains thousands of real-world job listings, encompassing diverse industries, roles, and compensation structures. Due to size constraints, the raw dataset is not included in this repository, but the heavily processed and cleaned versions are stored in the `data/processed/` directory.

## Core Features

1. **Task A: Skill Extractor (NER)**
   - Utilizes a fine-tuned JobBERT model (`jjzha/jobbert-base-cased`) on manually labeled datasets to perform Named Entity Recognition (NER).
   - Automatically reads job descriptions and extracts required **Hard Skills, Software Knowledge, and Certifications**.
   - Generates pseudo-labels for over 10,000 unannotated job postings, enabling large-scale skill profiling.

2. **Task B: Job Galaxy (Unsupervised Clustering)**
   - Converts textual job descriptions into dense embeddings using Sentence-BERT (`all-MiniLM-L6-v2`).
   - Reduces high-dimensional embedding spaces (384D) down to manageable 50D and 2D projections using **UMAP**.
   - Clusters the jobs using **HDBSCAN** to discover latent "Galaxies" of similar roles in the market, effectively mapping the entire job ecosystem visually.

3. **Task C: Salary Intelligence Simulator**
   - Combines NLP vectors (SBERT and TF-IDF) with structured, categorical data (Industry, Experience, Seniority).
   - Utilizes advanced Target Encoding for high-cardinality categorical features.
   - Uses a **Stacking Ensemble of XGBoost and LightGBM** (meta-learner: RidgeCV) to predict the fair market annual salary of a job posting.
   - Features an interactive **"What-If" simulator** in the dashboard to instantly see how tweaking variables (e.g., adding years of experience or shifting to a remote role) affects the predicted compensation.

## Development Team

- Rizki Piji Fathoni (24031554029)
- Alfin Jayadi (24031554082)
- Muhammad Rafi Fahrezi (24031554100)
- Daffa Ahmad Pangreksa (24031554159)

## Repository Structure

- `dashboard/`: Contains the interactive Streamlit application (`app.py`) for data visualization and the What-If Salary Simulator.
- `data/`: Contains processed and cleaned datasets used for training and dashboard visualization. (Raw datasets are excluded due to size limitations).
- `models/`: Stores the trained Target Encoder and Stacking Ensemble Models (`stacking.joblib`) used by the dashboard.
- `figures/`: Contains exported graphs and visualizations generated during the Exploratory Data Analysis (EDA) and clustering processes.
- `029_082_100_159_Project_ML.ipynb`: The main Jupyter Notebook detailing the entire data science workflow, model training, and evaluation.

## How to Run the Dashboard

1. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

2. Navigate to the `dashboard` directory:
   ```bash
   cd dashboard
   ```

3. Run the Streamlit application:
   ```bash
   streamlit run app.py
   ```

4. Access the dashboard in your browser at `http://localhost:8501`.

## Technologies Used

- **Deep Learning & NLP**: Hugging Face Transformers, Sentence-BERT, JobBERT
- **Machine Learning**: Scikit-Learn, XGBoost, LightGBM, HDBSCAN, UMAP
- **Web Dashboard**: Streamlit
- **Data Manipulation**: Pandas, NumPy
