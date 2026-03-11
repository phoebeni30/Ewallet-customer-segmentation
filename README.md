# 📊 Customer Segmentation Analysis
> **Uncovering Behavioral Archetypes in the Vietnamese F&B Ecosystem**

## 📑 Outline
1. [Problem Statement](#1-problem-statement)
2. [Project Description](#2-project-description)
3. [Analytical Pipeline](#3-analytical-pipeline)
4. [Dataset Characteristics](#4-dataset-characteristics)
5. [Project Architecture](#5-project-architecture)
6. [Results & Strategic Summary](#6-results--strategic-summary)
7. [Interactive Dashboard & Visualization](#7-interactive-dashboard--visualization)
8. [Conclusion & Future Work](#8-conclusion--future-work)

---

## 1. Problem Statement
In the modern competitive landscape, a **"one-size-fits-all"** marketing strategy is increasingly ineffective. Businesses often struggle to move beyond basic demographics to capture the true essence of their customers. 

Effective segmentation requires a multi-dimensional analysis across four fundamental pillars:

| Pillar | Focus | Significance |
| :--- | :--- | :--- |
| **Demographic** | Who | Identity (Age, Gender, Occupation, Marital Status). |
| **Geographic** | Where | Context and Environment (Location, Urban/Rural). |
| **Behavioral** | How | Observable Actions (Spending, Frequency, Usage patterns). |
| **Psychographic** | Why | Latent Drivers (Lifestyle, Social Status, Personality). |

**The Challenge:**
Most traditional methods fail to synthesize Behavioral and Psychographic aspects into actionable insights. This project addresses the problem of classifying customers into distinct, homogeneous groups based on their transaction footprints using **K-Means** and **Dimensionality Reduction (PCA)**.

---

## 2. Project Description

### 🌟 Overview
The Customer Segmentation Project is a specialized application of Machine Learning within **Business Intelligence** and **Marketing Analytics**. By analyzing granular transactional data from a leading Vietnamese E-wallet service, this project identifies distinct shopping patterns and behavioral archetypes.

### 🎯 Objectives
* **Data Synthesis:** Construct a comprehensive customer-level feature set from fragmented transactional logs.
* **Methodological Comparison:** Evaluate the trade-offs between manual RFM rule-based segmentation and K-Means algorithmic clustering.
* **Behavioral Deep-Dive:** Implement an expansive feature engineering pipeline to capture latent traits beyond simple spending.
* **Actionable Intelligence:** Deliver data-driven insights to optimize Marketing strategies and CRM.

---

## 3. Analytical Pipeline

1. **Preprocessing:** Data merging, exhaustive Cleaning, and Exploratory Data Analysis (EDA).
2. **Feature Engineering:** Transformation of transactional records into structured Customer Profiles.
3. **Segmentation Frameworks:**
    * **Task 1 (RFM Analysis):**
        * *Approach A (Manual):* Deterministic logic based on RFM scores.
        * *Approach B (K-Means):* RFM Extraction → Yeo-Johnson Transformation → Clustering.
    * **Task 2 (Advanced Behavioral Clustering):** High-dimensional Feature Engineering → PCA → K-Means Clustering.
4. **Synthesis:** Strategic Insight Generation and Business Recommendations.

---

## 4. Dataset Characteristics

### 📂 Sources & Scope
* **Source:** Transaction logs from a prominent Vietnamese E-wallet.
* **Tables:** `fact.csv`, `dim_merchant.csv`, `dim_store.csv`.
* **Sector:** Fintech & F&B (Food and Beverage).
* **Period:** Oct 01, 2021 – Jan 09, 2023.
* **Scale:** 1,048,575 Transactions | 583,618 Unique Users.

### 📊 Raw Data Structures

#### `fact.csv`
| Column Name | Data Type | Definition |
| :--- | :--- | :--- |
| **transID** | int64 | Unique identifier for each transaction. |
| **userID** | int64 | Unique identifier for the customer. |
| **Channel** | object | Consumption mode (Delivery, Dine In, Take Away). |
| **OrderFrom** | object | Order source (APP, STORE, WEBSITE, CALL CENTER). |
| **SalesAmount** | int64 | Total monetary value of the transaction. |
| **VoucherStatus** | object | Flag for promotion application (Yes/No). |

#### `dim_store.csv`
| Column Name | Data Type | Definition |
| :--- | :--- | :--- |
| **storeID** | object | Unique identifier for each physical merchant outlet or restaurant location. |
| **Province** | object | The administrative region or city (e.g., 'Ho Chi Minh', 'Hanoi') where the store is situated. |

#### `dim_merchant.csv`
| Column Name | Data Type | Definition |
| :--- | :--- | :--- |
| **appid** | int64 | Unique identifier for the specific application or platform version. |
| **merchantID** | int64 | Unique identifier for the business partner or merchant entity. |
| **merchantName** | object | The descriptive name of the merchant (e.g., brand name or restaurant name). |

### ⚠️ Data Challenges & Opportunities
* **Challenges:** High-value "Whales" skewing means; High proportion of "One-time Walk-ins".
* **Opportunities:** Categorical depth (App vs. Store preference); Advanced ratios (AOV, Voucher Reliance).

---

## 5. Project Architecture
```text
├── data/                    # Raw data
├── analysis/                # Jupyter Notebooks for EDA & Modeling
├── utils/                   # Helper functions and configurations
│   ├── cluster_model.py     # Clustering logic & PCA
│   ├── extract_customer_table.py   # Data merging & extraction
│   ├── rfm_manual.py        # Manual RFM pipeline
│   ├── rfm_feature_engineering.py  # RFM ML pipeline
│   ├── custom_features_config2.py  # Features Dictionary
│   └── custom_feature_engineering2.py # Behavioral pipeline
├── output/               # Processed and final data
├── dashboard/               # Exported Power BI dashboards & Screenshots
├── README.md                # Project documentation
└── requirements.txt         # Python libraries
```

---

## 6. Results & Strategic Summary

### 📊 Segment Profiles & Tactical Roadmap
| Cluster | Archetype | Value Contribution | Core Strategy | Primary Action |
| :--- | :--- | :--- | :--- | :--- |
| **Cluster 0** | **Casual Walk-ins** | Low Retention | **Conversion** | In-store discounts & Newcomer combos. |
| **Cluster 1** | **In-store Loyalists** | **2nd Highest** | **Experience** | Group sets & QR-table ordering. |
| **Cluster 2** | **Tech-Savviers** | **Highest** | **LTV Expansion** | VIP Tier & App Gamification. |
| **Cluster 3** | **Delivery Spenders** | High-Value | **Retention** | Office combos & Peak-hour Push notifications. |

### 💡 Segment Deep-Dive
* **Cluster 1 (Dine-in Seekers):** Highest Dine-in rate. They value the physical store as a "Third Place." Strategy focuses on QR-ordering and family-sized sets.
* **Cluster 2 & 3 (Digital Power Users):** Highly sensitive to vouchers. Strategy focuses on **Conditional Promotions** (e.g., "$5 off for orders over $20") to increase basket size.

---

## 7. Interactive Dashboard & Visualization
> **Tools:** Power BI Desktop, DAX, Field Parameters.

The dashboard provides a dynamic environment to monitor segment performance across various KPIs.

### 🖼️ Dashboard Preview
<p align="center">
  <img src="https://ibb.co/FbjMrGzS" width="850" alt="Customer Segmentation Dashboard">
</p>

### 🛠️ Key Features
* **Dynamic Metric Switching:** Using **Field Parameters** to toggle between **Sales Volume** and **Transaction Count**.
* **Period-over-Period (PoP):** Automated DAX to track MoM growth optimized for the F&B calendar.
* **Drill-down Analytics:** Interactive filters for Cluster ID, Order Source, and Geographic regions.

---

## 8. Conclusion & Future Work
### 🔑 Key Takeaways
* **Beyond RFM:** Incorporating categorical channels (App, Store, Web) captured nuances that simple RFM ignores.
* **Technical Critique:** 3D PCA explained **77%** variance. However, K-means faced challenges with overlapping "noise" from one-time users.

### 🚀 Future Improvements
* **Algorithmic Shift:** Experiment with **DBSCAN** or **GMM** to better handle non-spherical clusters.
* **Feature Enrichment:** Add time-based features (peak hours) and external data (holidays) for lifestyle mapping.
