# 📊 Customer Segmentation Analysis
> **Uncovering Behavioral Archetypes in the Vietnamese F&B Ecosystem**

## 📑 Outline
1. [Problem Statement](#1-problem-statement)
2. [Project Description](#2-project-description)
3. [Analytical Pipeline](#3-analytical-pipeline)
4. [Dataset Characteristics](#4-dataset-characteristics)
5. [Project Architecture](#5-project-architecture)

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
Most traditional methods fail to synthesize Behavioral and Psychographic aspects into actionable insights. This project addresses the problem of classifying customers into distinct, homogeneous groups based on their transaction footprints and interaction habits using **K-Means** and **Dimensionality Reduction (PCA)**.

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

1.  **Preprocessing:** Data merging, exhaustive Cleaning, and Exploratory Data Analysis (EDA).
2.  **Feature Engineering:** Transformation of transactional records into structured Customer Profiles.
3.  **Segmentation Frameworks:**
    * **Task 1 (RFM Analysis):** * *Approach A (Manual):* Deterministic logic based on RFM scores.
        * *Approach B (K-Means):* RFM Extraction → Yeo-Johnson Transformation → Clustering.
    * **Task 2 (Advanced Behavioral Clustering):** High-dimensional Feature Engineering → PCA → K-Means Clustering.
4.  **Synthesis:** Strategic Insight Generation and Business Recommendations.

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

#### `dim_store.csv` & `dim_merchant.csv`
* **Geography:** Province/City (e.g., Ho Chi Minh, Hanoi).
* **Identity:** Merchant Name, Merchant ID, App ID.

### ⚠️ Data Challenges & Opportunities
* **Challenges:** * *Extreme Volatility:* High-value "Whales" skewing statistical means.
    * *Information Sparsity:* Limited quantitative metrics.
    * *Skewed Distribution:* High proportion of "One-time Walk-ins".
* **Opportunities:** * *Categorical Depth:* Detecting correlations between App vs. Call Center preferences.
    * *Feature Expansion:* Deriving advanced ratios (AOV, Recency, Voucher Reliance).

---

## 5. Project Architecture
```text
├── data/                   # Raw and Processed data
├── analysis/              # Jupyter Notebooks for EDA & Modeling
├── utils/                  # Helper functions and configurations
│   ├── cluster_model.py    # Clustering logic & PCA
│   └── extract_customer_table.py  # Extract merged dataframe and requested tables
│   └── rfm_manual.py        # RFM features manual extracting pipeline
│   └── rfm_feature_engineering.py  # RFM engineering pipeline
│   └── custom_features_config2.py  # Custom Features Dictionary
│   └── custom_feature_engineering2.py  # Feature engineering pipeline
├── dashboard/                # Exported Power BI dashboards
└── README.md               # Project documentation
└── requirements.txt            # Python libraries

---

## 6. Results & Strategic Summary

### 📊 Segment Profiles & Tactical Roadmap
The analysis identified 4 distinct customer archetypes. Below is a summary of their behavioral characteristics and the corresponding business strategies:

| Cluster | Archetype | Volume/Value Contribution | Core Strategy | Primary Action |
| :--- | :--- | :--- | :--- | :--- |
| **Cluster 0** | **Casual Walk-ins** | Moderate AOV / Low Retention | **Conversion** | In-store return discounts & Newcomer combos. |
| **Cluster 1** | **Traditional Loyalists** | **2nd Highest** Contributor | **Experience** | Group sets, QR-table ordering & personalized in-store services. |
| **Cluster 2** | **Tech-Savviers** | **Highest** Contributor | **LTV Expansion** | VIP Tier program, Gamification & Threshold-based vouchers. |
| **Cluster 3** | **Bulk Delivery Spenders** | High-Value Transactions | **Retention** | Corporate/Office combos & Push-notifications at peak hours. |

---

### 💡 Segment Deep-Dive

#### **Cluster 1: The Dine-in Experience Seekers**
* **Insights:** This group prioritizes the physical store as a "Third Place." They prefer ordering via Website/Call Center and have the **highest Dine-in rate** among all clusters.
* **Strategic Move:** Instead of forcing app adoption, enhance their offline comfort with "Order & Pay at Table" QR solutions and bulk family/friend combos to increase basket size.

#### **Cluster 2 & 3: The Digital & Delivery Power Users**
* **Insights:** Cluster 2 drives the most volume via App/Vouchers, while Cluster 3 focuses on high-ticket delivery orders.
* **Strategic Move:** Shift from generic discounts to **Conditional Promotions** (e.g., "$5 off for orders over $20") to push spending beyond their typical threshold while maintaining engagement via App gamification.

---

### 🔍 Analytical Performance Summary
While the segments are business-actionable, the modeling process revealed critical technical insights:
* **Dimensionality Reduction:** 3D PCA successfully explained **77%** of the total variance, capturing the major behavioral trends (App usage, Order Value, and Channel preference).
* **Clustering Stability:** K-means with $k=4$ provided the best balance between statistical granularity and business interpretability, though it highlighted significant boundary overlap in the "one-time walk-in" density zone.
