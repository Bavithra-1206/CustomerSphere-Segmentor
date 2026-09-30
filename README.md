# 🎯 CustomerIQ - Customer Segmentation using K-Means Clustering

An enterprise-grade Customer Segmentation and Persona Intelligence Platform built with **Python**, **Scikit-Learn**, **Pandas**, **Chart.js**, and **Plotly**.

Segments customers into distinct behavioral and demographic groups based on spending patterns, income, age, recency, frequency, and monetary metrics using **K-Means Clustering** with **Principal Component Analysis (PCA)**.

---

## 🌟 Key Features

- **Multi-Dataset Support**:
  - **Mall Customers Benchmark**: 200 customer profiles (`Age`, `Annual Income (k$)`, `Spending Score (1-100)`).
  - **E-Commerce RFM + Demographics**: 450 customer profiles (`Recency`, `Frequency`, `Monetary Value`, `Annual Income`, `Age`, `Region`, `Preferred Category`).
- **Automated K Optimization**:
  - **Elbow Method (WCSS vs K)** line plot.
  - **Silhouette Score Analysis** (evaluating cluster cohesion & separation for $K = 2 \dots 10$).
- **Dimensionality Reduction & Interactive Viz**:
  - **2D / 3D PCA Scatter Plot** with interactive rotation, zooming, hover tooltips, and centroid marker highlights.
  - **Cluster Means Bar Chart** comparing segment averages against overall population baseline.
- **Automated Business Persona Generator**:
  - Automatically translates mathematical cluster centroids into human-readable customer archetypes (e.g., *💎 VIP High Rollers*, *💼 High-Income Savers*, *⚡ Impulse Trend Seekers*, *🎯 Budget Bargain Hunters*).
  - Generates actionable marketing strategies, channel recommendations, and churn risk levels.
- **Real-Time Customer Classifier**:
  - Interactive simulator to input custom customer metrics and instantly predict cluster assignment and persona playbook.
- **Interactive Dark Glassmorphic Dashboard**:
  - Built with modern CSS design system, responsive grid, Chart.js, and Plotly.js.
- **Data Export**:
  - Download segmented customer dataset with cluster labels in CSV format.

---

## 📁 Repository Structure

```
c:/Projet/
├── data/
│   ├── mall_customers.csv                   # Classic benchmark dataset
│   └── ecommerce_rfm_demographics.csv       # Extended RFM + Demographic dataset
├── src/
│   ├── __init__.py
│   ├── data_loader.py                       # Data loading & preprocessing (Standard/MinMax scaler)
│   ├── kmeans_engine.py                     # K-Means clustering, WCSS, Silhouette & PCA engine
│   └── persona_generator.py                 # Dynamic persona mapping & strategy engine
├── static/
│   ├── css/
│   │   └── style.css                        # Glassmorphism dark CSS theme
│   └── js/
│       └── app.js                           # Dashboard controller (Plotly, Chart.js, API calls)
├── templates/
│   └── index.html                           # Main web app layout & tabs
├── app.py                                   # Python HTTP REST API server
├── generate_datasets.py                     # Synthetic dataset generator script
├── test_pipeline.py                         # End-to-end pipeline verification test
├── customer_segmentation_analysis.ipynb     # Jupyter Notebook analysis & EDA
└── README.md
```

---

## 🚀 Quick Start Guide

### 1. Generate Datasets
Run `generate_datasets.py` to generate sample datasets in the `data/` folder:
```bash
python generate_datasets.py
```

### 2. Run the Interactive Web Dashboard
Start the REST API server:
```bash
python app.py
```
Open your browser and navigate to:
```
http://localhost:5000
```

### 3. Run Pipeline Test (CLI)
```bash
python test_pipeline.py
```

---

## 🔌 REST API Reference

| Endpoint | Method | Description |
| :--- | :--- | :--- |
| `GET /api/datasets` | `GET` | List available pre-loaded customer datasets |
| `POST /api/load_data` | `POST` | Load dataset, inspect numerical features and stats |
| `POST /api/elbow_silhouette` | `POST` | Calculate WCSS and Silhouette scores for K=2..10 |
| `POST /api/cluster` | `POST` | Run K-Means with specified K, features, and scaling method |
| `POST /api/predict` | `POST` | Classify a new customer input into a cluster segment |

---

## 💡 Customer Persona Mapping Logic

| Persona Archetype | Demographics / Behavior | Targeted Marketing Strategy |
| :--- | :--- | :--- |
| **💎 VIP High Rollers** | High Income, High Spending Score | Exclusive VIP concierge, early access to flagship products, private events. |
| **💼 High-Income Savers** | High Income, Low Spending Score | Value-driven newsletters, ROI case studies, quality guarantees, premium utility goods. |
| **⚡ Trend Seekers** | Low/Mod Income, High Spending Score | Influencer campaigns, flash sales, Buy-Now-Pay-Later (BNPL), gamified loyalty points. |
| **🎯 Bargain Hunters** | Low Income, Low Spending Score | Automated clearance emails, discount bundles, free shipping threshold incentives. |
| **⚖️ Mainstream Shoppers** | Moderate Income, Moderate Spending Score | Standard seasonal promo nudges, cross-selling accessories, repeat buyer rewards. |

---

## 📊 Jupyter Notebook Analysis
Open `customer_segmentation_analysis.ipynb` in VS Code or Jupyter Lab for a detailed step-by-step walkthrough of:
- Feature distribution analysis
- StandardScaler preprocessing
- WCSS elbow curve & silhouette plots
- 2D PCA scatter plots
- Segment profiling tables

---

## 📄 License
MIT License. Built for advanced customer analytics and machine learning applications.
