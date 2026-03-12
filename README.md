# 🛒 E-Commerce Revenue & Retention Intelligence Analysis

> End-to-end data analytics and data science project using the [Olist Brazilian E-Commerce dataset](https://www.kaggle.com/datasets/olistbr/brazilian-ecommerce/data) — from SQL data wrangling to ML-based churn prediction.

![Python](https://img.shields.io/badge/Python-3.12-blue)
![Pandas](https://img.shields.io/badge/Pandas-2.0+-green)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-1.8-orange)
![Status](https://img.shields.io/badge/Status-Complete-brightgreen)

---

## 📌 Project Overview

**Objective:** Analyze Olist's e-commerce operations to uncover revenue drivers, customer behavior patterns, and retention challenges — then apply ML models to predict customer churn.

**Key Finding:** 97% of customers purchase only once and never return. Traditional churn prediction fails (ROC-AUC ~0.54) because the marketplace business model provides no loyalty signal — a critical insight that redirects strategy from ML-based retention toward operational improvements.

### Tech Stack

| Tool | Purpose |
|------|---------|
| Python (Pandas, NumPy) | Data manipulation & analysis |
| SQL (SQLite) | Data joining & aggregation |
| Matplotlib / Seaborn | Visualization |
| Scikit-Learn | ML modeling (Logistic Regression, Random Forest, Gradient Boosting) |
| SciPy | Statistical hypothesis testing |
| Jupyter Notebook | Interactive analysis environment |

---

## 📊 Dataset

The Olist dataset contains **99,441 orders** from a Brazilian e-commerce marketplace (Sept 2016 – Oct 2018), spanning 9 relational tables:

| Table | Rows | Description |
|-------|------|-------------|
| orders | 99,441 | Central order table with timestamps & status |
| order_items | 112,650 | Items per order with price & freight |
| customers | 99,441 | Customer location (96,096 unique customers) |
| products | 32,951 | Product catalog with categories & dimensions |
| payments | 103,886 | Payment method, installments, value |
| reviews | 99,224 | Customer review scores (1–5) |
| sellers | 3,095 | Seller information |
| geolocation | 1,000,163 | Brazilian zip code coordinates |
| translations | 71 | Category name Portuguese → English |

---

## 🏗️ Project Structure

```
ecommerce-intelligence/
├── data/
│   ├── raw/                    # 9 original CSV files
│   └── processed/
│       └── rfm_scored.csv      # RFM table with segments & clusters
├── notebooks/
│   └── analysis.ipynb          # Main analysis notebook (62 cells)
├── reports/
│   └── figures/                # All saved visualizations (20+ charts)
├── sql/
│   └── master_query.sql        # Optimized SQL join
├── requirements.txt
├── README.md
└── EXECUTIVE_SUMMARY.md
```

---

## 📈 Analysis Pipeline

### Step 1: Data Profiling
- Data dictionary for all 9 tables
- Entity Relationship Diagram (ERD)
- Missing value analysis (reviews: 21% missing comments)
- Duplicate detection (geolocation: 261K duplicates)

### Step 2: SQL Master Query & Data Cleaning
- **Critical fix:** Resolved payment table join that caused row multiplication (items × payments = inflated revenue)
- Two-stage approach: Master query (6 tables) + pre-aggregated payments
- Derived metrics: `delivery_days`, `delivery_delay`, `item_revenue`

### Step 3: Revenue EDA
- Monthly revenue trend: Growth from R$4K to R$1.16M (290x in 14 months)
- Peak: November 2017 (Black Friday)
- Top state: São Paulo (37.4% of revenue)
- Top category: health_beauty (R$1.4M)
- Heatmap: Peak orders Mon–Wed, 10:00–16:00
- Payment: Credit card dominates (74.5%), Boleto (20.3%)

### Step 4: RFM Customer Segmentation

**Rule-Based RFM (11 segments):**

| Segment | Customers | % Revenue | Action |
|---------|-----------|-----------|--------|
| Loyal Customers | 26,580 (28.5%) | 40.5% | Reward program |
| Champions | 6,458 (6.9%) | 13.5% | VIP treatment |
| Potential Loyalists | 16,474 (17.6%) | 10.6% | Nurture to loyalty |
| Can't Lose Them | 4,238 (4.5%) | 8.9% | Win-back campaign |
| New Customers | 7,479 (8.0%) | 7.8% | Onboarding series |

**K-Means Clustering:**
- K=2 (Silhouette=0.69): Confirms "one-time" vs "repeat" as natural split
- K=4 (Silhouette=0.36): Further subdivides but with less statistical clarity
- Conclusion: Rule-Based provides actionable segments; K-Means validates the structure

### Step 5: Cohort & Retention Analysis
- **Retention crisis:** Month 1 retention = 0.48%, Month 12 = 0.21%
- Repeat purchase rate: Only 3% (2,801 out of 93,358)
- Repeat buyers return every ~71 days (median)
- 55.6% of repeat purchases happen same day (multi-order behavior)

### Step 6: Funnel & Conversion
- Funnel is strong: 97% delivery rate, 99.3% review rate
- Cancellation rate: Only 0.6% (625 orders)
- Delivery impact: Late delivery drops review from 4.2 to 2.3 stars
- Seller analysis: 3,095 sellers, median revenue R$1,039

### Step 7: Churn Prediction (ML)

| Model | ROC-AUC | Approach |
|-------|---------|----------|
| Logistic Regression | 0.63 | All customers |
| Random Forest | 0.55 | All customers |
| Gradient Boosting | 0.54 | All customers |

**Why models fail:** 99.4% churn rate creates extreme class imbalance. No distinguishable behavioral signal between the 3% who return and 97% who don't. This is inherent to Olist's marketplace model — customers don't choose Olist specifically.

**Portfolio insight:** Recognizing when ML is not the right tool is as valuable as building models.

### Step 9: Statistical Testing

| Hypothesis | Test | Result | Effect Size |
|------------|------|--------|-------------|
| Late delivery → Lower review | Welch's t-test | **Significant** (p<0.001) | d=1.38 (very large) |
| Repeat buyers → Higher AOV | Mann-Whitney U | Not significant (p=0.50) | d=-0.13 |
| Review differs by region | ANOVA | **Significant** (p<0.001) | F=52.18 |
| Higher price → Longer delivery | Kruskal-Wallis | **Significant** (p<0.001) | ρ=0.115 (weak) |

**Surprise finding:** Repeat buyers do NOT spend more per order (R$125 vs R$147). Their value comes from frequency, not basket size.

---

## 🔑 Key Insights

### 1. The Repeat Customer Crisis
97% of customers buy once and never return. This is Olist's #1 challenge, but it's structural — as a B2B marketplace, customers have no direct relationship with the Olist brand.

### 2. Delivery is the Biggest Lever
Late delivery causes a 2-star drop in reviews (4.2 → 2.3, Cohen's d=1.38). This is the most actionable finding: improving logistics directly improves customer satisfaction.

### 3. ML Churn Prediction Has Limits
With 99.4% churn rate, no ML model can meaningfully predict who will return. Rule-based segmentation (RFM) provides more business value than sophisticated ML for this use case.

### 4. Regional Satisfaction Gap
Northeast and North regions score significantly lower (3.91–3.97) vs Southeast (4.11), likely driven by longer delivery distances.

### 5. Revenue Concentration
São Paulo alone generates 37.4% of revenue. Top 5 states = 72.2% (Pareto principle). Geographic expansion has significant upside.

---

## 🚀 Recommendations

1. **Delivery optimization:** Reduce late delivery rate (currently 6.6%) — highest ROI action
2. **Post-purchase engagement:** Email at Day 7 (thank you), Day 30 (cross-sell), Day 71 (win-back)
3. **Seller quality program:** Monitor high late-rate sellers, incentivize fast shipping
4. **Regional strategy:** Improve logistics infrastructure for Northeast/North
5. **First-order maximization:** Since most customers won't return, maximize first-order value through cross-sell at checkout

---

## 📂 How to Run

```bash
# Clone the repository
git clone https://github.com/yourusername/ecommerce-intelligence.git
cd ecommerce-intelligence

# Install dependencies
pip install -r requirements.txt

# Download dataset from Kaggle
# Place CSV files in data/raw/

# Open notebook
jupyter notebook notebooks/analysis.ipynb
```

### Requirements
```
pandas>=2.0
numpy>=1.24
matplotlib>=3.7
seaborn>=0.13
scikit-learn>=1.3
scipy>=1.11
squarify>=0.4
```

---

## 👤 Author

**Perawit** — Data Analyst / Business Analyst  
Portfolio project demonstrating end-to-end analytics workflow from SQL to ML.

---

## 📄 License

This project uses the [Olist Brazilian E-Commerce Public Dataset](https://www.kaggle.com/datasets/olistbr/brazilian-ecommerce/data) under CC BY-NC-SA 4.0.
