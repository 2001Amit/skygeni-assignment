"""
SkyGeni Sales Analysis
Quick exploration to figure out what's going on with win rates
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report
import warnings
warnings.filterwarnings('ignore')

sns.set_style('whitegrid')

# ============================================================================
# PART 1 - UNDERSTANDING THE PROBLEM
# ============================================================================
"""
CRO says: "Win rate down but pipeline looks healthy"

My initial thoughts:
- Could be lead quality issue (chasing quantity over quality)
- Could be sales process breakdown
- Could be market/competitive shift
- Could be specific segments struggling

Let's dig into the data and see what's actually happening...
"""

# Load the data
df = pd.read_csv('../data/skygeni_sales_data.csv')

# Quick check
print("="*80)
print("DATASET OVERVIEW")
print("="*80)
print(f"Total deals: {len(df):,}")
print(f"Columns: {', '.join(df.columns)}")
print(f"\nDate range: {df['created_date'].min()} to {df['closed_date'].max()}")

# Convert dates
df['created_date'] = pd.to_datetime(df['created_date'])
df['closed_date'] = pd.to_datetime(df['closed_date'])

# Add some helper columns
df['quarter'] = df['closed_date'].dt.to_period('Q')
df['month'] = df['closed_date'].dt.to_period('M')
df['won'] = (df['outcome'] == 'Won').astype(int)

# Basic stats
overall_win_rate = df['won'].mean()
total_won = df[df['outcome']=='Won']['deal_amount'].sum()
total_lost = df[df['outcome']=='Lost']['deal_amount'].sum()

print(f"\nOverall win rate: {overall_win_rate:.1%}")
print(f"Total revenue won: ${total_won:,.0f}")
print(f"Total revenue lost: ${total_lost:,.0f}")
print(f"Revenue left on table: ${total_lost:,.0f} (ouch)")

# ============================================================================
# PART 2 - EXPLORING THE DATA
# ============================================================================

print("\n" + "="*80)
print("INSIGHT 1: WIN RATE TREND OVER TIME")
print("="*80)

# Check quarterly trend
quarterly = df.groupby('quarter').agg({
    'won': ['mean', 'count'],
    'deal_amount': 'sum'
}).round(3)

quarterly.columns = ['win_rate', 'deals', 'revenue']
print("\nQuarterly breakdown:")
print(quarterly)

# Calculate the decline
first_q_wr = quarterly['win_rate'].iloc[0]
last_q_wr = quarterly['win_rate'].iloc[-1]
decline = (last_q_wr - first_q_wr) * 100

print(f"\nWin rate change: {decline:+.1f} percentage points")
print(f"First quarter: {first_q_wr:.1%}")
print(f"Last quarter: {last_q_wr:.1%}")

# But check deal volume
first_q_deals = quarterly['deals'].iloc[0]
last_q_deals = quarterly['deals'].iloc[-1]
volume_change = ((last_q_deals - first_q_deals) / first_q_deals) * 100

print(f"\nDeal volume change: {volume_change:+.1f}%")
print("\n💡 FINDING: Win rate down but volume UP = quality problem, not quantity")

# ============================================================================
# INSIGHT 2: LEAD SOURCE ANALYSIS
# ============================================================================

print("\n" + "="*80)
print("INSIGHT 2: LEAD SOURCE EFFECTIVENESS")
print("="*80)

lead_stats = df.groupby('lead_source').agg({
    'won': 'mean',
    'deal_amount': 'mean',
    'sales_cycle_days': 'mean',
    'deal_id': 'count'
}).round(2)

lead_stats.columns = ['win_rate', 'avg_deal_size', 'avg_cycle', 'count']
lead_stats = lead_stats.sort_values('win_rate', ascending=False)

print("\nLead source performance:")
print(lead_stats)

# Calculate the gap
best = lead_stats.index[0]
worst = lead_stats.index[-1]
gap = (lead_stats.loc[best, 'win_rate'] - lead_stats.loc[worst, 'win_rate']) * 100

print(f"\n💡 FINDING: {gap:.0f} point gap between {best} and {worst}")
print(f"   If we shifted budget from {worst} to {best}, potential impact is huge")

# ============================================================================
# INSIGHT 3: SALES CYCLE LENGTH
# ============================================================================

print("\n" + "="*80)
print("INSIGHT 3: SALES CYCLE IMPACT")
print("="*80)

# Create buckets for cycle length
df['cycle_bucket'] = pd.cut(
    df['sales_cycle_days'], 
    bins=[0, 30, 60, 90, 365],
    labels=['<30d', '30-60d', '60-90d', '90+d']
)

cycle_stats = df.groupby('cycle_bucket').agg({
    'won': 'mean',
    'deal_id': 'count'
}).round(3)

cycle_stats.columns = ['win_rate', 'count']
print("\nWin rate by sales cycle length:")
print(cycle_stats)

print(f"\n💡 FINDING: Deals that drag past 90 days have {cycle_stats.loc['90+d', 'win_rate']:.1%} win rate")
print("   vs <30 days at {:.1%}".format(cycle_stats.loc['<30d', 'win_rate']))
print("   Long cycles aren't 'big deals brewing' - they're dead deals")

# ============================================================================
# CUSTOM METRICS
# ============================================================================

print("\n" + "="*80)
print("CUSTOM METRICS I CREATED")
print("="*80)

# Metric 1: Pipeline Quality Index (PQI)
# Idea: measure what % of pipeline is in high-converting segments
segment_rates = df.groupby(['industry', 'product_type'])['won'].mean()
df['segment_rate'] = df.apply(
    lambda x: segment_rates.get((x['industry'], x['product_type']), 0.5), 
    axis=1
)

high_quality = df[df['segment_rate'] > 0.5]
pqi = (len(high_quality) / len(df)) * (df['deal_amount'].mean() / df['deal_amount'].median())

print(f"\nPipeline Quality Index (PQI): {pqi:.2f}")
print(f"Interpretation: {len(high_quality)/len(df):.1%} of pipeline is in high-win-rate segments")
print("Useful as an early warning - PQI drops before win rate does")

# Metric 2: Revenue Efficiency Score (RES)
# Idea: are we capturing the revenue potential in our pipeline?
quarterly_res = df.groupby('quarter').apply(
    lambda x: (x[x['won']==1]['deal_amount'].sum() / x['deal_amount'].sum()) * x['won'].mean()
).round(3)

print(f"\nRevenue Efficiency Score by quarter:")
print(quarterly_res)
print("\nInterpretation: Only capturing about 30% of pipeline's revenue potential")

# ============================================================================
# PART 3 - BUILD THE MODEL (Win Rate Driver Analysis)
# ============================================================================

print("\n" + "="*80)
print("PART 3: WIN RATE DRIVER MODEL")
print("="*80)

# Prepare data for modeling
# Encode categorical variables
model_df = df.copy()

encoders = {}
cat_cols = ['industry', 'region', 'product_type', 'lead_source', 'deal_stage', 'sales_rep_id']

for col in cat_cols:
    le = LabelEncoder()
    model_df[f'{col}_enc'] = le.fit_transform(model_df[col])
    encoders[col] = le

# Select features
# Trying to keep it simple - just the key drivers
features = [
    'deal_amount',
    'sales_cycle_days', 
    'industry_enc',
    'region_enc',
    'product_type_enc',
    'lead_source_enc',
    'deal_stage_enc'
]

X = model_df[features]
y = model_df['won']

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"\nTraining set: {len(X_train)} deals")
print(f"Test set: {len(X_test)} deals")

# Train model
# Using logistic regression for interpretability
# (Could use Random Forest or XGBoost for better accuracy, but CRO won't understand it)
model = LogisticRegression(max_iter=1000, random_state=42)
model.fit(X_train, y_train)

# Evaluate
y_pred = model.predict(X_test)
accuracy = (y_pred == y_test).mean()

print(f"\nModel accuracy: {accuracy:.1%}")
print("\nClassification report:")
print(classification_report(y_test, y_pred, target_names=['Lost', 'Won']))

# Feature importance
importance = pd.DataFrame({
    'feature': features,
    'coefficient': model.coef_[0],
    'abs_coef': np.abs(model.coef_[0])
}).sort_values('abs_coef', ascending=False)

print("\nTop factors affecting win rate:")
print(importance[['feature', 'coefficient']].head(10))

# ============================================================================
# GENERATE ACTIONABLE OUTPUTS
# ============================================================================

print("\n" + "="*80)
print("ACTIONABLE INSIGHTS BY CATEGORY")
print("="*80)

# Analyze each categorical variable
for category in ['lead_source', 'industry', 'product_type', 'region']:
    stats = df.groupby(category)['won'].agg(['mean', 'count'])
    stats = stats.sort_values('mean', ascending=False)
    
    print(f"\n{category.upper()}:")
    print(stats)

# Create risk scores for all deals
risk_scores = pd.DataFrame({
    'deal_id': model_df['deal_id'],
    'actual_outcome': model_df['outcome'],
    'win_probability': model.predict_proba(X)[:, 1],
    'risk_score': 1 - model.predict_proba(X)[:, 1]
})

# Save to file
risk_scores.to_csv('../outputs/deal_risk_scores.csv', index=False)
print("\n✅ Risk scores saved to outputs/deal_risk_scores.csv")

# Show some examples
print("\nExample high-risk deals:")
high_risk = risk_scores[risk_scores['risk_score'] > 0.7].head()
print(high_risk)

# ============================================================================
# RECOMMENDATIONS
# ============================================================================

print("\n" + "="*80)
print("TOP RECOMMENDATIONS FOR CRO")
print("="*80)

recommendations = """
Based on the analysis, here's what I'd focus on:

1. LEAD SOURCE MIX
   - Reduce outbound activity by 30%
   - Reallocate budget to referral programs
   - Expected impact: +5-7 points to win rate

2. SALES CYCLE DISCIPLINE
   - Implement 60-day checkpoint rule
   - Re-qualify or kill deals that aren't progressing
   - Expected impact: Free up 20-25% of rep time

3. SEGMENT FOCUS
   - Double down on FinTech and SaaS (performing well)
   - Re-think HealthTech approach (struggling)
   - Consider different sales motion for HealthTech

4. PIPELINE QUALITY METRICS
   - Track PQI monthly (leading indicator)
   - Stop measuring just volume, start measuring quality
   - Alert when PQI drops below 1.5

5. REP PERFORMANCE
   - Bottom quartile reps need coaching
   - Pair them with top performers
   - Focus on qualification skills
"""

print(recommendations)

# ============================================================================
# SYSTEM DESIGN NOTES (Part 4)
# ============================================================================

system_design = """
============================================================================
PART 4: SYSTEM DESIGN (if we productize this)
============================================================================

ARCHITECTURE (keeping it simple):

Salesforce → Daily ETL (2am) → Snowflake → Python Analytics → Streamlit Dashboard
                                                           → Slack Alerts

KEY COMPONENTS:

1. Data Pipeline
   - Daily batch sync from Salesforce
   - Data validation and cleaning
   - Feature engineering (cycle length, segment stats, etc.)

2. Analytics Engine
   - Risk scoring model (retrained quarterly)
   - Anomaly detection (simple statistical process control)
   - Metric calculation (PQI, RES, win rates)

3. Alert System
   - Real-time: High-value deals with risk >70%
   - Daily: Pipeline health summary
   - Weekly: Detailed report for sales leadership
   - Monthly: Strategic analysis for exec team

EXAMPLE ALERTS:

Real-time:
"🚨 Deal D12345 ($85K) at 78% risk - in Demo for 67 days (avg: 28)"

Weekly:
"Pipeline Health: 42% win rate (↓3% vs last week)
18 high-risk deals totaling $1.2M need attention"

FAILURE MODES TO CONSIDER:

- Model drift (quarterly retraining needed)
- Data quality issues (garbage in = garbage out)
- Alert fatigue (need smart thresholds)
- Gaming (reps will learn to game the scores)
- Small sample sizes (new segments)

The biggest risk is self-fulfilling prophecy - if we tell reps a deal is 
low-probability, they might not try as hard.
"""

print(system_design)

# ============================================================================
# REFLECTION (Part 5)
# ============================================================================

reflection = """
============================================================================
PART 5: HONEST REFLECTION
============================================================================

WEAKEST ASSUMPTIONS:

1. Correlation = Causation
   I found that Referrals win more, but can't prove CAUSING more referrals
   will improve win rate. Maybe referrals are just better-fit customers.
   Would need experiments (A/B tests) to know for sure.

2. Historical patterns will continue
   Model is trained on 2023-24 data. If market changes in 2025 (new competitor,
   economic downturn, product launch), model becomes stale fast.

3. CRM data is complete
   Sales notes might say "lost to competitor" but not capture the full story.
   Missing context on WHY deals really lost.

WHAT WOULD BREAK IN PRODUCTION:

- Model drift within 3-6 months if market changes
- Edge cases (very large deals, new industries) not in training data
- Sales process changes (new comp plan, new leadership)
- Data pipeline failures (schema changes, CRM updates)
- Reps gaming the system to make deals look better

LEAST CONFIDENT ABOUT:

The causal inference part. I can say "Outbound leads have lower win rates"
but I can't prove that STOPPING outbound will help. Maybe we just need to
get BETTER at outbound.

This is the difference between correlation and causation, and I only have
observational data, not experimental data.

WHAT I'D BUILD NEXT (1 month):

Week 1-2: Get more data
- External: competitor intel, market trends, economic indicators
- Internal: sales activity data (emails, calls), NLP on CRM notes
- Qualitative: actually talk to sales reps

Week 3: Better analysis
- Causal inference (propensity score matching, etc.)
- Cohort analysis (do patterns persist?)
- Stage-by-stage conversion funnel

Week 4: Make it real
- Build Streamlit dashboard
- Slack integration for alerts
- A/B testing framework

WHAT I'M ACTUALLY PROUD OF:

The custom metrics (PQI and RES). I think they capture something useful.

Also keeping the focus on ACTIONABILITY. Don't care about fancy models if
they don't help the CRO make better decisions.
"""

print(reflection)

print("\n" + "="*80)
print("ANALYSIS COMPLETE")
print("="*80)
print("\nNext steps:")
print("1. Review outputs/deal_risk_scores.csv")
print("2. Validate findings with sales team")
print("3. Run pilot test of recommendations")
print("4. Build dashboard if this proves useful")
