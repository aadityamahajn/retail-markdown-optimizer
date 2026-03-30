# Retail Markdown Optimizer

A complete end-to-end project combining **Machine Learning (Demand Forecasting)** and **Operations Research (Markdown Optimization)** using the M5 Walmart dataset.

## Project Overview

This project demonstrates a full retail analytics pipeline:

- **Phase 1**: Data preparation & memory optimization on large M5 dataset
- **Phase 2**: Demand Forecasting with feature engineering, LightGBM, and recursive prediction
- **Phase 3**: Markdown Optimization using a production-ready Greedy Heuristic (after MILP failed due to weak price elasticity signal)

### Key Highlights from Documentation

- Built robust demand forecasting models (Naive → Linear Regression → LightGBM with Tweedie loss)
- Added hierarchical features, lag/rolling signals, and post-prediction calibration
- Developed a practical Greedy Heuristic for markdown decisions under budget constraint
- Achieved 100% volume calibration and 100% budget utilization in optimization

**Key Takeaway**: In real retail, when data lacks strong price elasticity, a well-designed greedy heuristic often outperforms complex MILP solvers in speed, adoption, and business impact.

## Repository Structure
retail-markdown-app/
├── app.py                          ← Streamlit Web App
├── models/                         ← 10 trained LightGBM models (per store)
├── sample_simulation.csv
├── sample_sales.csv
├── notebooks/                       ← Original Colab Notebooks
└── README.md

## How to Run the Streamlit App
1. Clone the repository
2. Install dependencies:
   ```bash
   pip install streamlit pandas joblib matplotlib seaborn
   pip install lightgbm          # Required to load the saved models
