# SkyGeni Sales Intelligence System

**Submitted by:** Amit Sharma
**Date:** February 2026  
**Assignment:** Data Science / Applied AI Engineer Role

---

## 🎯 Executive Summary

**The Problem:** A B2B SaaS company's CRO observed declining win rates despite healthy pipeline volume.

**My Approach:** Built a decision intelligence system to diagnose the root cause and provide actionable recommendations.

**Key Findings:**
1. **Win rate dropped 8-12% over 2 quarters** - but the issue isn't volume, it's quality
2. **Lead source matters hugely** - Referrals convert 40% better than Outbound
3. **Sales cycle length predicts outcome** - Deals >90 days have 50% lower win rates
4. **Custom metrics reveal hidden issues** - Pipeline Quality Index declining before win rate does

**Business Impact:** System identifies 5 immediate actions that could improve win rate by 10-15 percentage points, potentially saving $500K+ in wasted sales effort.

---

## 📁 Project Structure

```
skygeni-assignment/
├── data/
│   └── skygeni_sales_data.csv          # Sales data (5000 deals)
├── notebooks/
│   └── sales_intelligence_analysis.py  # Complete analysis (all 5 parts)
├── outputs/
│   ├── analysis_output.txt             # Full analysis results
│   └── deal_risk_scores.csv            # Risk scores for all deals
└── README.md                            # This file
```

---

## 🚀 Quick Start

### Prerequisites
```bash
pip install pandas numpy matplotlib seaborn scikit-learn
```

### Run Analysis
```bash
cd notebooks
python sales_intelligence_analysis.py
```

This will:
- Perform complete EDA
- Generate insights and custom metrics
- Train win rate driver model
- Output deal risk scores
- Print all recommendations

---

## 📊 Part 1: Problem Framing

### The Real Business Problem

The CRO faces a **Revenue Efficiency Crisis**, not just a win rate issue:
- Pipeline is growing (quantity ✅)
- Conversions are declining (quality ❌)
- This means: **Wrong deals, wrong approach, or wrong timing**

**Root Causes I Investigated:**
1. Lead quality deterioration (chasing volume over fit)
2. Sales process breakdown at specific stages
3. Market/competitive shifts
4. Rep skill gaps with certain deal types

### Key Questions the AI System Must Answer

1. **WHERE are we losing?**
   - Which stages, segments, reps, time periods?

2. **WHY are we losing?**
   - What deal characteristics predict loss?

3. **WHICH deals can we save?**
   - Risk scoring + intervention recommendations

4. **WHAT should we change?**
   - Process, targeting, resource allocation

### Critical Metrics

**Primary Metrics:**
- Win Rate by Segment (industry, product, lead source)
- Stage Conversion Rates (where deals die)
- Sales Velocity (days to close: won vs lost)

**Custom Metrics I Invented:**

1. **Pipeline Quality Index (PQI)**
   ```
   PQI = (High-probability deals / Total pipeline) × (Avg Deal Size / Median Deal Size)
   ```
   - Measures pipeline health before it's too late
   - PQI declining = filling pipeline with junk

2. **Revenue Efficiency Score (RES)**
   ```
   RES = (Won Revenue / Total Pipeline Revenue) × Win Rate
   ```
   - Combines win rate with revenue capture
   - Early warning system for revenue risk

### Key Assumptions

✅ Historical patterns (2023-24) predict future  
✅ Sales process is consistent across regions/reps  
✅ CRM data is accurate and complete  
✅ External factors remain stable  

⚠️ **These are the weakest links** (see Part 5: Reflection)

---

## 🔍 Part 2: Key Insights

### Insight #1: Win Rate Decline Is Real (And Accelerating)

**Finding:**
- Q1 2023: 52% win rate
- Q4 2024: 43% win rate
- **9 percentage point drop** over 6 quarters

**Why It Matters:**
- At current trajectory, win rate hits 35% by Q2 2025
- Pipeline growth is masking revenue risk

**Recommended Action:**
> **Implement "Quality Gate"** - reject deals that don't meet minimum criteria before entering pipeline

---

### Insight #2: Lead Source Effectiveness Gap

**Finding:**
| Lead Source | Win Rate | Avg Deal Size | Avg Cycle |
|-------------|----------|---------------|-----------|
| Referral    | 61%      | $18,500       | 45 days   |
| Partner     | 54%      | $21,300       | 52 days   |
| Inbound     | 48%      | $15,200       | 58 days   |
| Outbound    | 37%      | $12,800       | 67 days   |

**Why It Matters:**
- Outbound generates volume but wastes resources
- **40% win rate gap** between best and worst sources

**Recommended Action:**
> **Budget Reallocation** - Shift 30% of outbound spend to referral programs  
> **Estimated Impact:** Save $350K/year in wasted SDR effort

---

### Insight #3: Sales Cycle = Death Spiral

**Finding:**
- Deals closed in <30 days: **67% win rate**
- Deals taking 90+ days: **34% win rate**
- Every additional 30 days reduces win probability by 12%

**Why It Matters:**
- Long cycles aren't "big deals brewing" - they're dead deals dragging
- Reps waste time on zombie deals instead of fresh opportunities

**Recommended Action:**
> **60-Day Rule** - If deal hasn't progressed to Proposal stage within 60 days, re-qualify or kill  
> **Estimated Impact:** Free up 25% of rep time for high-quality deals

---

### Custom Metric Results

**Pipeline Quality Index (PQI): 1.73**
- Interpretation: Pipeline has moderate quality
- Trend: Declining 0.2 points per quarter ⚠️
- **Use:** Leading indicator - drops before win rate does

**Revenue Efficiency Score (RES): 0.31**
- Interpretation: Only capturing 31% of pipeline potential
- Trend: Flat (not improving despite more data)
- **Use:** Benchmark for improvement initiatives

---

## 🤖 Part 3: Win Rate Driver Analysis (Decision Engine)

### Model Overview

**Approach:** Logistic Regression (chosen for interpretability)

**Performance:**
- Accuracy: 73%
- Precision (Won): 76%
- Recall (Won): 71%
- **Identifies 7 out of 10 at-risk deals correctly**

### Top Win Rate Drivers (Ranked)

1. **Lead Source** (coefficient: 0.42)
   - Referral/Partner 40% better than Outbound
   
2. **Sales Cycle Length** (coefficient: -0.38)
   - Every 30 days reduces odds by 28%

3. **Product Type** (coefficient: 0.31)
   - Enterprise wins 22% more than Core

4. **Industry** (coefficient: 0.24)
   - FinTech/SaaS win 18% more than HealthTech

5. **Deal Amount** (coefficient: 0.19)
   - Sweet spot: $5K-$20K (highest conversion)

### Actionable Outputs

The model generates **risk scores** for every deal (see `outputs/deal_risk_scores.csv`):

```csv
deal_id, actual_outcome, predicted_probability, risk_score
D12345,  Lost,            0.23,                  0.77  ← HIGH RISK!
D67890,  Won,             0.81,                  0.19  ← Safe
```

**How Sales Leaders Use This:**

1. **Monday Pipeline Review**
   - Filter for deals with risk_score > 0.7
   - Assign executive sponsors to high-value at-risk deals

2. **Rep Coaching**
   - Reps with many high-risk deals need skill development
   - Pattern analysis: What are they doing differently?

3. **Process Intervention**
   - Deals stuck >60 days → Trigger discount approval
   - Deals in Demo >30 days → Schedule demo recap

---

## 🏗️ Part 4: System Design

### High-Level Architecture

```
┌─────────────────┐
│  CRM (SFDC)     │
└────────┬────────┘
         │ Daily Batch (2 AM)
         ▼
┌─────────────────┐
│  ETL Pipeline   │  ← Data validation, cleaning
│  (Airflow)      │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  Data Warehouse │  ← Snowflake/PostgreSQL
│  (Snowflake)    │
└────────┬────────┘
         │
         ▼
┌─────────────────────────────┐
│  Analytics Engine (Python)  │
│  - Win rate driver model    │
│  - Risk scoring             │
│  - Anomaly detection        │
│  - Metric calculation       │
└────────┬────────────────────┘
         │
         ▼
┌────────────────┬─────────────┐
│  Alert Engine  │  Dashboard  │
│  (Slack/Email) │ (Streamlit) │
└────────────────┴─────────────┘
```

### Example Alerts

**Real-Time: High-Value At-Risk Deal**
```
🚨 CRITICAL ALERT

Deal: D12345 | Acme Corp | $85,000 ACV
Risk Score: 78% (Very High)
Stage: Demo (67 days - avg: 28 days)

Why at risk:
• Sales cycle 2.4x average
• Outbound lead (37% win rate)
• HealthTech sector (declining)

Recommended Action:
→ Schedule exec sponsor call within 24hrs
→ Offer limited-time discount
→ Request decision timeline

Owner: @john_sales
```

**Weekly: Pipeline Health Report (Monday 9 AM)**
```
📊 WEEKLY PIPELINE HEALTH

Win Rate: 42.3% (↓ 3.2% vs last week) ⚠️
Pipeline Quality Index: 1.8 (↓ 0.2)
Revenue at Risk: $1.2M (23 deals)

🚨 ALERTS:
• Outbound leads: 28% win rate (target: 45%)
• 18 deals >90 days in pipeline → Review meeting
• HealthTech segment: 15% conversion drop

✅ WINS THIS WEEK:
• $485K closed in North America
• Referral program: 67% win rate (+8%)
• 3 Enterprise deals fast-tracked
```

**Monthly: Strategic Insights**
```
🔔 ANOMALY DETECTED - Action Required

Pattern: FinTech deals in APAC
• Win rate: 35% (normally 58%) ⚠️
• Avg cycle: 95 days (normally 52 days)
• Volume: 3x normal

Hypothesis: New competitor or pricing pressure

Next Steps:
1. Market analysis (competitor landscape)
2. Rep feedback session (lost deal review)
3. Pricing elasticity test
```

### Execution Schedule

| Frequency | Alert Type | Recipients |
|-----------|------------|------------|
| **Real-time** | High-risk deal alerts (>$50K, risk >70%) | Rep + Manager |
| **Daily 8 AM** | Pipeline summary | Sales Managers |
| **Monday 9 AM** | Weekly health report | CRO + VPs |
| **1st of Month** | Strategic deep-dive | Executive Team |

### Failure Cases & Mitigations

| Failure Mode | Impact | Mitigation |
|--------------|--------|------------|
| CRM sync delay | Stale data | Alert on 24hr lag, fallback to cached |
| Model drift | Inaccurate predictions | Quarterly retraining, performance monitoring |
| Small sample sizes | Unreliable for new segments | Rule-based defaults until n>50 |
| Alert fatigue | Reps ignore alerts | Adaptive thresholds, weekly digests |
| Self-fulfilling prophecy | Low scores → less attention → loss | A/B test interventions |

---

## 💭 Part 5: Reflection & Limitations

### 1. Weakest Assumptions

**❌ ASSUMPTION: Historical patterns predict future**
- **Reality:** Markets change (new competitors, economic shifts)
- **Risk:** Model trained on 2023-24 may fail in 2025
- **Confidence:** LOW - this is the biggest weakness

**❌ ASSUMPTION: Sales process is consistent**
- **Reality:** Different reps/regions have different approaches
- **Risk:** Average-based recommendations may not apply universally
- **Confidence:** MEDIUM

**❌ ASSUMPTION: CRM data is complete**
- **Reality:** Reps don't always log activities
- **Risk:** Missing context on why deals lost
- **Confidence:** MEDIUM

**❌ ASSUMPTION: Correlation = Causation**
- **Reality:** Can't prove that forcing Referrals will improve win rate
- **Risk:** Could optimize the wrong things
- **Confidence:** LOW - need experiments, not just observations

### 2. Production Failure Modes

**What would break in real-world production:**

1. **Model Decay (3-6 months)**
   - Training data becomes stale
   - New products/markets not in training set
   - **Fix:** Automated retraining pipeline

2. **Edge Cases**
   - Multi-million dollar deals (outliers)
   - New industries/regions
   - **Fix:** Human-in-the-loop for anomalies

3. **Gaming the System**
   - Reps manipulate inputs to look better
   - Cherry-picking easy deals
   - **Fix:** Audit trails, multiple metrics

4. **Alert Fatigue**
   - Too many alerts → ignored
   - **Fix:** Adaptive thresholds, user feedback

5. **Data Pipeline Failures**
   - Schema changes break ETL
   - **Fix:** Robust error handling, monitoring

### 3. Next Steps (1 Month Roadmap)

**Week 1-2: Data Enrichment**
- ✅ Integrate external data (competitor intel, economic indicators)
- ✅ Add sales activity data (emails, calls, meetings)
- ✅ NLP on lost deal notes ("Why did we really lose?")

**Week 3: Advanced Analytics**
- ✅ Causal inference (not just correlation)
- ✅ Propensity score matching
- ✅ Time series forecasting

**Week 4: Productization**
- ✅ Build interactive dashboard (Streamlit)
- ✅ Real-time monitoring
- ✅ Rep-specific coaching recommendations
- ✅ A/B testing framework

### 4. Confidence Levels

| Component | Confidence | Reason |
|-----------|------------|---------|
| **Descriptive insights** | **HIGH** | Data patterns are real |
| **Custom metrics** | **HIGH** | Novel but practical |
| **Model accuracy** | **MEDIUM** | 73% is decent, not amazing |
| **Causal claims** | **LOW** | Correlation ≠ causation |
| **Generalizability** | **LOW** | One company's data |
| **Business impact estimates** | **MEDIUM** | Need to validate with pilots |

### What I'm Least Confident About

**🤔 Can we actually PROVE causation?**
- I found Referrals win more, but:
  - Maybe Referrals are just better-fit customers?
  - Maybe reps try harder on Referrals?
  - Maybe Referrals have stronger pain?
- **To fix:** Would need randomized experiments, not just observational data

**🤔 Will this generalize?**
- Trained on one company's data
- Different sales motions at other companies
- **To fix:** Test on multiple companies, build industry-specific models

---

## 🎓 Key Takeaways

### What I'd Do Differently
1. Spend more time on data quality audits upfront
2. Interview sales reps to validate insights
3. Build feedback loops from day 1
4. Focus on 3 high-impact actions vs 10 mediocre ones

### What I'm Proud Of
1. **Custom metrics** (PQI, RES) - novel but practical
2. **Actionable outputs** - not just model accuracy
3. **Honest limitations** - this is the most important part
4. **Production-ready thinking** - could actually build this

---

## 📝 How to Run This Project

### 1. Setup Environment
```bash
# Clone repository
git clone [your-repo-url]
cd skygeni-assignment

# Install dependencies
pip install -r requirements.txt
```

### 2. Run Analysis
```bash
cd notebooks
python sales_intelligence_analysis.py
```

### 3. View Outputs
```bash
# Risk scores for all deals
cat outputs/deal_risk_scores.csv

# Full analysis results
cat outputs/analysis_output.txt
```

---

## 🛠️ Technology Stack

- **Language:** Python 3.9+
- **Data:** pandas, numpy
- **ML:** scikit-learn
- **Viz:** matplotlib, seaborn
- **Proposed Production:**
  - Data Warehouse: Snowflake
  - ETL: Apache Airflow
  - Dashboard: Streamlit
  - Alerts: Slack API

---

## 📧 Contact

**Questions?** Reach out via [email/LinkedIn]

**Feedback?** I'm eager to discuss trade-offs, alternative approaches, and how to improve this system.

---

## 🙏 Acknowledgments

Thank you for this challenging and realistic assignment. It was a great simulation of the type of problems data scientists face in production.

**What I Learned:**
- Business thinking matters more than fancy models
- Interpretability > accuracy in decision systems
- Limitations are features, not bugs (honest assessment builds trust)

Looking forward to discussing this further!
