# Job Market Intelligence: Multi-Task NLP Pipeline

This repository contains the final project for the Machine Learning course (Even Semester 2025/2026). It implements a comprehensive Multi-Task Natural Language Processing (NLP) pipeline designed to extract insights, cluster roles, and predict salaries from raw job postings.

## Project Vision & Impact

In the modern job market, raw job descriptions contain massive amounts of unstructured insights that are difficult to analyze at scale. This project tackles that challenge by building a Multi-Task NLP Pipeline. Instead of relying on simple keyword matching, we utilize state-of-the-art Deep Learning models to truly understand the context of job postings.

## Core Features

1. **Task A: Skill Extractor (NER)**
   - Utilizes a fine-tuned JobBERT model (`jjzha/jobbert-base-cased`) on manually labeled datasets to perform Named Entity Recognition (NER).
   - Automatically reads job descriptions and extracts required Hard Skills, Software Knowledge, and Certifications.

2. **Task B: Job Galaxy (Unsupervised Clustering)**
   - Converts textual job descriptions into dense vectors using Sentence-BERT (`all-MiniLM-L6-v2`).
   - Reduces dimensionality using UMAP and clusters the jobs using HDBSCAN to discover latent "Galaxies" of similar roles in the market.

3. **Task C: Salary Intelligence Simulator**
   - Combines NLP vectors with structured data (Industry, Experience, Seniority).
   - Uses an XGBoost ensemble model to predict the fair market annual salary of a job posting.
   - Provides an interactive "What-If" simulator to instantly see how tweaking variables affects predicted salaries.

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
   pip install streamlit pandas numpy joblib scikit-learn category-encoders xgboost lightgbm
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
