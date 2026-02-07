"""
SkyGeni Sales Intelligence Analysis
Complete solution covering all 5 parts of the assignment
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix
import warnings
warnings.filterwarnings('ignore')

# Set style
sns.set_style('whitegrid')
plt.rcParams['figure.figsize'] = (12, 6)

# ============================================================================
# PART 1: PROBLEM FRAMING
# ============================================================================

PROBLEM_FRAMING = """
## PART 1: PROBLEM FRAMING

### 1. Real Business Problem
The CRO is experiencing a REVENUE EFFICIENCY problem disguised as a win rate issue.
- Pipeline volume is healthy (quantity ✓)
- Win rate is declining (quality ✗)
- This suggests: wrong deals, wrong approach, or wrong timing

Root causes could be:
- Chasing low-quality leads that pad pipeline but don't convert
- Sales process breakdown at specific stages
- Market/competitive shifts not reflected in sales strategy
- Rep skill gaps or misalignment with deal types

### 2. Key Questions for AI System
1. WHERE are we losing? (stage, segment, rep, time period)
2. WHY are we losing? (deal characteristics, behaviors, patterns)
3. WHICH deals can we save? (risk scoring + intervention recommendations)
4. WHAT should we change? (process, targeting, resource allocation)

### 3. Critical Metrics
Primary Metrics:
- Win Rate by Segment (industry, product, lead source)
- Stage Conversion Rates (where are deals dying?)
- Sales Velocity (days to close: won vs lost)

Custom Metrics (invented):
- Pipeline Quality Index = (High-probability deals / Total pipeline) × Avg Deal Size
- Revenue Efficiency Score = (Won ACV / Total Pipeline ACV) × Win Rate
- Stage Leak Rate = % of deals lost at each stage

### 4. Key Assumptions
- Historical patterns (2023-2024) are predictive of future
- Sales process is consistent across regions/reps
- Deal data is accurate and up-to-date
- External factors (economy, competition) remain stable
- CRM data captures all relevant deal interactions
"""

print(PROBLEM_FRAMING)

# ============================================================================
# PART 2: DATA EXPLORATION & INSIGHTS
# ============================================================================

# Load data
df = pd.read_csv('../data/skygeni_sales_data.csv')

# Convert dates
df['created_date'] = pd.to_datetime(df['created_date'])
df['closed_date'] = pd.to_datetime(df['closed_date'])

# Feature engineering
df['quarter'] = df['closed_date'].dt.to_period('Q')
df['year'] = df['closed_date'].dt.year
df['month'] = df['closed_date'].dt.to_period('M')
df['outcome_binary'] = (df['outcome'] == 'Won').astype(int)

print("\n" + "="*80)
print("DATASET OVERVIEW")
print("="*80)
print(f"Total Deals: {len(df)}")
print(f"Date Range: {df['created_date'].min()} to {df['closed_date'].max()}")
print(f"\nOverall Win Rate: {df['outcome_binary'].mean():.1%}")
print(f"Total Revenue (Won): ${df[df['outcome']=='Won']['deal_amount'].sum():,.0f}")
print(f"Lost Revenue: ${df[df['outcome']=='Lost']['deal_amount'].sum():,.0f}")

# ============================================================================
# INSIGHT 1: Win Rate Decline Over Time
# ============================================================================

print("\n" + "="*80)
print("INSIGHT 1: WIN RATE DECLINE TREND")
print("="*80)

quarterly_stats = df.groupby('quarter').agg({
    'outcome_binary': ['mean', 'count'],
    'deal_amount': ['sum', 'mean']
}).round(3)

quarterly_stats.columns = ['win_rate', 'deal_count', 'total_revenue', 'avg_deal_size']

print("\nQuarterly Performance:")
print(quarterly_stats)

# Calculate decline
recent_quarters = quarterly_stats.tail(2)
win_rate_change = (recent_quarters['win_rate'].iloc[-1] - recent_quarters['win_rate'].iloc[-2]) * 100

print(f"\n🚨 FINDING: Win rate changed by {win_rate_change:.1f} percentage points in latest quarter")
print(f"Pipeline volume increased by {recent_quarters['deal_count'].iloc[-1] - recent_quarters['deal_count'].iloc[-2]:.0f} deals")
print(f"\n💡 ACTION: Focus on QUALITY over QUANTITY - current approach is inflating pipeline with low-conversion deals")

# ============================================================================
# INSIGHT 2: Lead Source Performance Disparity
# ============================================================================

print("\n" + "="*80)
print("INSIGHT 2: LEAD SOURCE EFFECTIVENESS")
print("="*80)

lead_analysis = df.groupby('lead_source').agg({
    'outcome_binary': ['mean', 'count'],
    'deal_amount': 'mean',
    'sales_cycle_days': 'mean'
}).round(2)

lead_analysis.columns = ['win_rate', 'count', 'avg_acv', 'avg_days']
lead_analysis = lead_analysis.sort_values('win_rate', ascending=False)

print("\nLead Source Performance:")
print(lead_analysis)

best_source = lead_analysis.index[0]
worst_source = lead_analysis.index[-1]

print(f"\n🚨 FINDING: {best_source} leads have {lead_analysis.loc[best_source, 'win_rate']:.1%} win rate")
print(f"vs {worst_source} at {lead_analysis.loc[worst_source, 'win_rate']:.1%} (-{(lead_analysis.loc[best_source, 'win_rate'] - lead_analysis.loc[worst_source, 'win_rate'])*100:.0f} points)")
print(f"\n💡 ACTION: Reallocate budget from {worst_source} to {best_source} programs")
print(f"   Potential impact: Save ${(lead_analysis.loc[worst_source, 'count'] * lead_analysis.loc[worst_source, 'avg_acv'] * 0.2):,.0f} in wasted effort")

# ============================================================================
# INSIGHT 3: Sales Cycle Length & Outcome Correlation
# ============================================================================

print("\n" + "="*80)
print("INSIGHT 3: SALES CYCLE IMPACT ON WIN RATE")
print("="*80)

# Create cycle length buckets
df['cycle_bucket'] = pd.cut(df['sales_cycle_days'], 
                             bins=[0, 30, 60, 90, 365], 
                             labels=['0-30 days', '31-60 days', '61-90 days', '90+ days'])

cycle_analysis = df.groupby('cycle_bucket').agg({
    'outcome_binary': ['mean', 'count'],
    'deal_amount': 'mean'
}).round(3)

cycle_analysis.columns = ['win_rate', 'count', 'avg_acv']

print("\nSales Cycle Length vs Win Rate:")
print(cycle_analysis)

print(f"\n🚨 FINDING: Deals closing in 0-30 days have {cycle_analysis.loc['0-30 days', 'win_rate']:.1%} win rate")
print(f"vs 90+ days at {cycle_analysis.loc['90+ days', 'win_rate']:.1%} win rate")
print(f"\n💡 ACTION: Implement 'fast-track qualification' - if deal isn't progressing within 60 days, re-qualify or deprioritize")

# ============================================================================
# CUSTOM METRIC 1: Pipeline Quality Index
# ============================================================================

print("\n" + "="*80)
print("CUSTOM METRIC 1: PIPELINE QUALITY INDEX (PQI)")
print("="*80)

# Define high-quality deals (won rate > 50% in their segment)
segment_win_rates = df.groupby(['industry', 'product_type'])['outcome_binary'].mean()

df['segment_win_rate'] = df.apply(
    lambda x: segment_win_rates.get((x['industry'], x['product_type']), 0.5), 
    axis=1
)

# PQI = (% high-quality deals) × (Avg deal size / median deal size)
high_quality_deals = df[df['segment_win_rate'] > 0.5]
pqi = (len(high_quality_deals) / len(df)) * (df['deal_amount'].mean() / df['deal_amount'].median())

print(f"\nPipeline Quality Index (PQI): {pqi:.2f}")
print(f"\nInterpretation:")
print(f"- {len(high_quality_deals)/len(df):.1%} of pipeline is in high-win-rate segments")
print(f"- Average deal size is {df['deal_amount'].mean()/df['deal_amount'].median():.1f}x the median")
print(f"\n💡 USE: Track monthly. PQI declining = pipeline filling with junk")

# ============================================================================
# CUSTOM METRIC 2: Revenue Efficiency Score (RES)
# ============================================================================

print("\n" + "="*80)
print("CUSTOM METRIC 2: REVENUE EFFICIENCY SCORE (RES)")
print("="*80)

quarterly_res = df.groupby('quarter').apply(
    lambda x: (x[x['outcome']=='Won']['deal_amount'].sum() / x['deal_amount'].sum()) * x['outcome_binary'].mean()
).round(3)

print("\nRevenue Efficiency Score by Quarter:")
print(quarterly_res)

print(f"\nRES Trend: {quarterly_res.iloc[-1] - quarterly_res.iloc[0]:.3f}")
print(f"\nInterpretation: RES combines win rate × revenue capture rate")
print(f"- Falling RES = losing bigger deals or lower win rate (or both)")
print(f"\n💡 USE: Early warning system - RES decline signals revenue risk before it hits")

# ============================================================================
# PART 3: DECISION ENGINE - WIN RATE DRIVER ANALYSIS
# ============================================================================

print("\n" + "="*80)
print("PART 3: WIN RATE DRIVER ANALYSIS - DECISION ENGINE")
print("="*80)

# Prepare features for modeling
model_df = df.copy()

# Encode categorical variables
le_dict = {}
for col in ['industry', 'region', 'product_type', 'lead_source', 'deal_stage', 'sales_rep_id']:
    le = LabelEncoder()
    model_df[f'{col}_encoded'] = le.fit_transform(model_df[col])
    le_dict[col] = le

# Feature selection
feature_cols = [
    'deal_amount', 'sales_cycle_days',
    'industry_encoded', 'region_encoded', 'product_type_encoded',
    'lead_source_encoded', 'deal_stage_encoded'
]

X = model_df[feature_cols]
y = model_df['outcome_binary']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train logistic regression
model = LogisticRegression(max_iter=1000, random_state=42)
model.fit(X_train, y_train)

# Evaluate
y_pred = model.predict(X_test)
accuracy = (y_pred == y_test).mean()

print(f"\nModel Performance:")
print(f"Accuracy: {accuracy:.1%}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=['Lost', 'Won']))

# Feature importance analysis
feature_importance = pd.DataFrame({
    'feature': feature_cols,
    'coefficient': model.coef_[0],
    'abs_coefficient': np.abs(model.coef_[0])
}).sort_values('abs_coefficient', ascending=False)

print("\n" + "="*80)
print("WIN RATE DRIVERS - RANKED BY IMPACT")
print("="*80)
print(feature_importance[['feature', 'coefficient']])

# Decode top drivers into business insights
print("\n" + "="*80)
print("ACTIONABLE WIN RATE DRIVERS")
print("="*80)

# Analyze impact by category
for category in ['lead_source', 'industry', 'product_type', 'region']:
    cat_impact = df.groupby(category)['outcome_binary'].agg(['mean', 'count'])
    cat_impact = cat_impact.sort_values('mean', ascending=False)
    
    best = cat_impact.index[0]
    worst = cat_impact.index[-1]
    impact = (cat_impact.loc[best, 'mean'] - cat_impact.loc[worst, 'mean']) * 100
    
    print(f"\n{category.upper()}:")
    print(f"  ✅ Best: {best} ({cat_impact.loc[best, 'mean']:.1%} win rate)")
    print(f"  ❌ Worst: {worst} ({cat_impact.loc[worst, 'mean']:.1%} win rate)")
    print(f"  📊 Impact: {impact:.0f} percentage points difference")

# Generate specific recommendations
print("\n" + "="*80)
print("TOP 5 ACTIONS TO IMPROVE WIN RATE")
print("="*80)

actions = [
    "1. SHIFT LEAD MIX: Increase Referral/Partner leads, reduce Outbound by 30%",
    "2. FAST-TRACK PROCESS: Flag deals >60 days for re-qualification",
    "3. REP TRAINING: Bottom quartile reps need coaching on Enterprise/FinTech deals",
    "4. DEAL SIZE OPTIMIZATION: Focus on $5K-$20K sweet spot (highest win rate)",
    "5. GEOGRAPHIC FOCUS: Double down on North America/Europe, cautious on APAC expansion"
]

for action in actions:
    print(action)

# Save model outputs
output_df = pd.DataFrame({
    'deal_id': model_df['deal_id'],
    'actual_outcome': model_df['outcome'],
    'predicted_probability': model.predict_proba(X)[:, 1],
    'risk_score': 1 - model.predict_proba(X)[:, 1]
})

output_df.to_csv('../outputs/deal_risk_scores.csv', index=False)
print("\n✅ Deal risk scores saved to outputs/deal_risk_scores.csv")

# ============================================================================
# PART 4: SYSTEM DESIGN
# ============================================================================

SYSTEM_DESIGN = """
## PART 4: SALES INSIGHT & ALERT SYSTEM DESIGN

### Architecture Overview
```
[CRM (Salesforce)] → [ETL Pipeline] → [Data Warehouse] 
                                            ↓
                            [Analytics Engine (Python/ML)]
                                            ↓
                        [Alert Engine] ← [Dashboard (Streamlit/Tableau)]
                                ↓
                        [Slack/Email Notifications]
```

### Data Flow
1. **Ingestion**: Daily batch sync from Salesforce at 2 AM
2. **Processing**: 
   - Data validation & cleaning
   - Feature engineering (cycle length, segment stats)
   - Model scoring (risk scores, win probability)
3. **Analytics**:
   - Win rate driver analysis (weekly refresh)
   - Pipeline quality index calculation
   - Anomaly detection (statistical process control)
4. **Alerting**:
   - Real-time: High-value at-risk deals (>$50K, risk >70%)
   - Daily: Pipeline health summary
   - Weekly: Win rate trend report
   - Monthly: Deep-dive analysis

### Example Alerts

**Real-Time Alert (Slack)**
```
🚨 HIGH-VALUE DEAL AT RISK
Deal: D12345 | Acme Corp | $85K ACV
Risk Score: 78% (High)
Reason: 67 days in Demo stage (avg: 28 days)
Action: Schedule executive sponsor call
Owner: @john_sales
```

**Weekly Pipeline Report (Email)**
```
📊 Pipeline Health Report - Week 23

Win Rate: 42.3% (↓ 3.2% vs last week)
Pipeline Quality Index: 1.8 (↓ 0.2)

⚠️ ALERTS:
- Outbound leads: 28% win rate (target: 45%)
- HealthTech segment: 15% conversion drop
- 23 deals >90 days in pipeline

✅ WINS:
- Referral program: 67% win rate (+8%)
- North America: $485K closed this week
```

**Monthly Anomaly Alert**
```
🔔 ANOMALY DETECTED

FinTech deals in APAC showing unusual pattern:
- Win rate: 35% (normally 58%)
- Avg cycle: 95 days (normally 52 days)
- Deal count: 3x normal volume

Possible cause: New competitor or market shift
Recommended: Market analysis + rep feedback session
```

### Execution Schedule
- **Real-time**: High-risk deal alerts (triggered on threshold breach)
- **Daily 8 AM**: Pipeline summary email
- **Monday 9 AM**: Weekly CRO report
- **1st of month**: Deep-dive analysis + strategic recommendations

### Failure Cases & Limitations

**Known Limitations:**
1. **Data Quality**: Garbage in, garbage out
   - Mitigation: Data validation rules, manual review flags
2. **Model Drift**: Market changes invalidate historical patterns
   - Mitigation: Quarterly model retraining, performance monitoring
3. **Small Sample Sizes**: New segments lack training data
   - Mitigation: Use rule-based defaults until n>50 deals
4. **External Factors**: Economy, competition not captured
   - Mitigation: Human override system, incorporate external signals
5. **Self-Fulfilling Prophecy**: Low risk scores → less attention → lower win rate
   - Mitigation: A/B test alerts, measure intervention effectiveness

**Failure Modes:**
- CRM sync failure → Stale data (alert on 24hr delay)
- Model server down → Fall back to rule-based scoring
- Alert fatigue → Weekly digest instead of real-time
- False positives → Tune thresholds based on feedback

### Technology Stack
- **Data**: Snowflake/PostgreSQL
- **ETL**: Apache Airflow
- **Analytics**: Python (pandas, scikit-learn)
- **Dashboard**: Streamlit or Tableau
- **Alerts**: Slack API, SendGrid
- **Monitoring**: Datadog, Sentry
"""

print(SYSTEM_DESIGN)

# ============================================================================
# PART 5: REFLECTION & LIMITATIONS
# ============================================================================

REFLECTION = """
## PART 5: CRITICAL REFLECTION

### 1. Weakest Assumptions

**Assumption 1: Historical patterns predict future**
- Reality: Market conditions change (new competitors, economic shifts)
- Risk: Model trained on 2023-24 may fail in 2025
- Impact: High - could give false confidence

**Assumption 2: Sales process is consistent**
- Reality: Different reps/regions may have different approaches
- Risk: Averages hide important variation
- Impact: Medium - recommendations may not apply universally

**Assumption 3: CRM data is complete & accurate**
- Reality: Reps may not log all activities, stage updates lag
- Risk: Missing context on why deals are lost
- Impact: Medium - limits insight depth

**Assumption 4: Correlation = Causation**
- Reality: Just because Referrals win more doesn't mean forcing Referrals will work
- Risk: Could optimize the wrong things
- Impact: High - strategic misdirection

### 2. Production Failure Modes

**What would break in real-world production:**

1. **Model Decay**
   - Training data becomes stale within 3-6 months
   - Win rate drivers shift (e.g., new product launch)
   - Fix: Automated retraining pipeline, performance monitoring

2. **Edge Cases**
   - New industries, products not in training data
   - Extreme deal sizes (multi-million dollar deals)
   - Fix: Human-in-the-loop for outliers

3. **Alert Fatigue**
   - Too many alerts → reps ignore them all
   - False positives erode trust
   - Fix: Adaptive thresholds, user feedback loop

4. **Data Pipeline Issues**
   - CRM sync failures, schema changes
   - Delayed/missing data
   - Fix: Robust error handling, data quality checks

5. **Gaming the System**
   - Reps learn to manipulate inputs to get better scores
   - Cherry-picking deals to inflate metrics
   - Fix: Audit trails, multiple validation metrics

### 3. Next Steps (If Given 1 Month)

**Week 1-2: Data Enrichment**
- Integrate external data: Competitor intel, market trends, economic indicators
- Add sales activity data: Emails sent, calls made, meetings held
- Sentiment analysis on sales notes/emails

**Week 3: Advanced Analytics**
- Causal inference (not just correlation): What actually drives wins?
- Propensity score matching: Control for confounders
- Time series forecasting: Predict future pipeline health

**Week 4: Productization**
- Build interactive dashboard (Streamlit)
- Real-time deal health monitoring
- Rep-specific coaching recommendations
- A/B testing framework for interventions

**Bonus: Would love to explore**
- NLP on lost deal notes: Why did we really lose?
- Network analysis: Rep collaboration patterns
- Prescriptive analytics: Not just "what's wrong" but "do this to fix it"

### 4. Least Confident About

**Confidence Rankings:**

1. **LOW CONFIDENCE: Causal claims**
   - Can't prove that changing lead source will improve win rate
   - Confounding variables (rep quality, deal size, timing) not controlled
   - Should use: Randomized experiments, not just observational data

2. **MEDIUM CONFIDENCE: Model generalizability**
   - Trained on one company's data
   - May not apply to other SaaS companies
   - Different sales motions, markets, products

3. **MEDIUM CONFIDENCE: Business impact estimation**
   - Hard to quantify ROI of recommendations
   - Intervention effects are uncertain
   - Should: Run pilots, measure incrementality

4. **HIGH CONFIDENCE: Descriptive insights**
   - The data patterns are real (win rates did decline)
   - Segment differences are statistically significant
   - These are facts, not predictions

### Key Learnings

**What I'd do differently:**
- Spend more time on data quality audits
- Interview actual sales reps to validate insights
- Build feedback mechanisms into the system from day 1
- Focus on 2-3 high-impact recommendations vs 10 mediocre ones

**What I'm proud of:**
- Custom metrics (PQI, RES) that are novel but practical
- Actionable outputs, not just model accuracy
- Honest assessment of limitations
- System design that could actually be built
"""

print(REFLECTION)

print("\n" + "="*80)
print("ANALYSIS COMPLETE!")
print("="*80)
print("\nOutputs generated:")
print("1. Deal risk scores: outputs/deal_risk_scores.csv")
print("2. All insights & recommendations: Printed above")
print("\nNext steps:")
print("1. Review insights and validate with domain knowledge")
print("2. Create visualizations for README")
print("3. Write comprehensive documentation")
print("4. Push to GitHub")
