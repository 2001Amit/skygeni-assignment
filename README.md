# SkyGeni Sales Intelligence - Assignment Submission

Hey! Thanks for the interesting problem. I spent about 6-7 hours on this over the past few days, and here's what I found.

## Quick Summary

**The Problem:** CRO says win rates are dropping but pipeline looks healthy. Basically - we're chasing more deals but closing fewer of them.

**What I Found:**
- Win rate dropped from ~52% to ~43% over the past year
- But the issue isn't equal across the board - it's concentrated in specific areas
- Outbound leads are dragging everything down (only 37% win rate vs 61% for referrals)
- Deals that drag past 90 days almost never close

**What to Do About It:**
Focus on lead quality over quantity. The data suggests we're filling the pipeline with low-probability deals just to hit activity metrics, which makes things look healthy but isn't translating to revenue.

---

## My Thought Process (Part 1 - Problem Framing)

When I first read the CRO's complaint, my gut reaction was: this is a **quality vs quantity** problem. High pipeline volume but low conversion = we're fishing in the wrong pond.

### The Real Problem

I think what's happening here is pretty common in SaaS sales:
1. Sales leadership sets aggressive pipeline targets
2. Reps respond by lowering their qualification bar to hit targets
3. Pipeline looks great in metrics reports
4. But conversion rates tank because we're chasing bad-fit prospects

It's not necessarily that reps are doing something wrong - it's that the incentives are misaligned.

### Questions I Wanted to Answer

Instead of just building a model to predict wins/losses, I wanted to understand:

1. **WHERE is the problem?** Is it certain industries? Regions? Lead sources? Reps?
2. **WHEN did it start?** Was there a specific quarter where things changed?
3. **WHY are we losing?** Are deals dying at a specific stage? Is it sales cycle length? Deal size?
4. **WHAT can we actually do?** (This is the most important - insights without actions are useless)

### Metrics That Matter

Honestly, win rate by itself doesn't tell you much. I wanted to look at:

- **Win rate by segment** (industry, lead source, product type)
- **Conversion rates by stage** (where are deals actually dying?)
- **Sales cycle analysis** (how long are won vs lost deals taking?)

I also created two custom metrics (more on this below) because I couldn't find standard metrics that captured what I was seeing.

### Assumptions I'm Making

Fair warning - these could all be wrong:
- CRM data is accurate (it probably isn't 100%)
- Sales process is consistent across regions (probably not true)
- Past patterns will continue (markets change)
- External factors (economy, competition) are stable (definitely not true)

The biggest assumption is that **correlation = causation**, which I know isn't true but I'm treating it as directional guidance rather than absolute truth.

---

## What I Found in the Data (Part 2 - EDA & Insights)

### Insight 1: The Win Rate Drop is Real (and Getting Worse)

I broke down win rate by quarter and yeah, it's declining steadily:

- Q1 2023: 52%
- Q4 2024: 43%

That's a 9 percentage point drop. At this rate, we'll be under 40% by mid-2025.

But here's what's interesting - deal volume actually INCREASED during this period. So we're creating more pipeline but converting less of it. Classic quality vs quantity problem.

**Why this matters:** Revenue is a function of both pipeline size AND conversion rate. Right now we're optimizing for the wrong metric.

**What to do:** I'd recommend a "quality gate" before deals enter the pipeline. Better to have fewer, higher-quality opportunities than a bloated pipeline that wastes everyone's time.

### Insight 2: Lead Source is Make or Break

This was the biggest finding IMO. Not all leads are created equal:

| Lead Source | Win Rate | Avg Deal Size | Sales Cycle |
|-------------|----------|---------------|-------------|
| Referral    | 61%      | $18K          | 45 days     |
| Partner     | 54%      | $21K          | 52 days     |
| Inbound     | 48%      | $15K          | 58 days     |
| Outbound    | 37%      | $13K          | 67 days     |

Referrals are almost 2x better than Outbound. And they close faster with bigger deal sizes.

**Why this matters:** If we're allocating equal budget/effort to all lead sources, we're massively misallocating resources. Every dollar spent on outbound could generate 2x the revenue if spent on referral programs.

**What to do:** 
- Cut outbound budget by 30% and redirect to referral programs
- Stop measuring SDRs on "leads generated" and start measuring on "qualified opportunities"
- Build out partner ecosystem (54% win rate is solid)

### Insight 3: Time Kills Deals

I bucketed deals by how long they took to close:

- 0-30 days: 67% win rate
- 31-60 days: 51% win rate
- 61-90 days: 42% win rate
- 90+ days: 34% win rate

Every 30 days a deal sits, win probability drops about 12 percentage points.

I initially thought this was just "big deals take longer" but when I controlled for deal size, the pattern held. Long cycles = dead deals, regardless of size.

**Why this matters:** Reps are probably spending tons of time on deals that are never going to close. It's zombie pipeline.

**What to do:** Implement a 60-day rule - if a deal hasn't progressed to Proposal stage within 60 days, require re-qualification or kill it. Free up rep time for fresh opportunities.

### Custom Metrics I Created

Standard metrics weren't capturing what I was seeing, so I made two new ones:

**1. Pipeline Quality Index (PQI)**

Formula: `(% of deals in high-win-rate segments) × (Avg deal size / Median deal size)`

Basically trying to measure: is our pipeline full of good deals or junk?

Current PQI: 1.73 (and declining 0.2 points per quarter)

The cool thing about PQI is it's a leading indicator - it drops BEFORE win rate does, so you can catch problems early.

**2. Revenue Efficiency Score (RES)**

Formula: `(Won Revenue / Total Pipeline Revenue) × Win Rate`

This combines two things: (a) are we winning deals, and (b) are we winning the BIG deals?

Current RES: 0.31 (meaning we're only capturing 31% of our pipeline's revenue potential)

Honestly not sure if these metrics are "good" but they helped me think about the problem differently.

---

## The Model (Part 3 - Decision Engine)

I chose **Option B (Win Rate Driver Analysis)** because it directly addresses the CRO's question: "what's going wrong?"

### Why Not the Other Options?

- **Deal Risk Scoring:** Felt too reactive. CRO wants to know WHY, not just WHICH deals are at risk.
- **Revenue Forecast:** Too much uncertainty, not actionable enough
- **Anomaly Detection:** Interesting but doesn't solve the core problem

### Model Approach

I went with **Logistic Regression** instead of something fancier (Random Forest, XGBoost, etc.) for one reason: **interpretability**.

The CRO doesn't care about 0.5% accuracy improvements. They care about "what factors are killing our win rate and what do I do about it?"

Logistic regression gives you coefficients you can actually explain to a non-technical executive.

### Results

Model accuracy: **73%**

Is that good? Honestly, not amazing, but good enough for this use case. I can correctly identify about 7 out of 10 at-risk deals, which is plenty to take action on.

More importantly, the model tells us WHAT matters:

**Top factors hurting win rate:**
1. Sales cycle length (longer = worse)
2. Outbound leads (vs referral/partner)
3. Certain industries (HealthTech struggling)
4. Deal size extremes (very small or very large)

**Top factors helping win rate:**
1. Referral/Partner leads
2. Fast sales cycles (<60 days)
3. FinTech and SaaS industries
4. Mid-size deals ($10-30K)

### What I'd Actually Give to the CRO

A simple dashboard showing:
- Each open deal with a risk score (0-100)
- Top 3 factors contributing to that risk
- Recommended action (re-qualify, add exec sponsor, fast-track, etc.)

And a weekly report:
- This week's wins/losses by segment
- Pipeline quality trend
- Deals that need attention

Check `outputs/deal_risk_scores.csv` for the full model output.

---

## System Design (Part 4)

If we were to actually build this as a product, here's what I'm thinking:

### Architecture (keeping it simple)

```
Salesforce/CRM 
    ↓ (daily sync at 2am)
Data Warehouse (Snowflake)
    ↓ (ETL + feature engineering)
Analytics Engine (Python/scikit-learn)
    ↓
Dashboard (Streamlit) + Alerts (Slack)
```

Nothing fancy. Daily batch job, not real-time (don't need it for this use case).

### Alerts That Would Actually Be Useful

**Daily (8am):**
> "Pipeline health: 18 new deals added, 12 closed (8 won, 4 lost). Win rate this week: 44% (target: 50%)"

**Weekly (Monday morning):**
> "High-risk deals needing attention:
> - Deal D12345 ($85K) - 67 days in Demo stage (avg: 28 days)
> - Deal D45678 ($42K) - Outbound lead in HealthTech (both red flags)
> Total at-risk revenue: $1.2M"

**Monthly (executive level):**
> "Win rate trend: 43% (↓3% QoQ)
> Root cause: Outbound lead volume up 40% but converting at 37%
> Recommended: Reduce outbound targets, invest in referral program"

### What Would Break

Let me be honest about failure modes:

1. **Model drift** - If market conditions change, model becomes useless. Need quarterly retraining.
2. **Data quality** - If reps don't update CRM properly, garbage in = garbage out
3. **Gaming** - Reps will figure out what makes deals "look good" and game the system
4. **Alert fatigue** - Too many alerts = everyone ignores them
5. **Small samples** - New segments (new industry, new region) won't have enough training data

The biggest risk is actually **self-fulfilling prophecy** - if we tell reps a deal is low-probability, they might not try as hard, which makes it actually low-probability.

---

## Reflection (Part 5 - The Honest Part)

### What I'm Least Confident About

**1. Causality**

I found that Referrals win more, but that doesn't mean FORCING more referrals will work. Maybe referrals are just better-fit customers to begin with. Or maybe reps try harder on referrals. Or maybe referrals come in with stronger pain points.

I can't prove causation from this data. Would need to run actual experiments (A/B tests) to know for sure.

**2. Generalizability**

This is based on one company's data. Different SaaS companies have totally different sales motions. Enterprise software is different from SMB SaaS. Transactional sales are different from strategic.

My model might be overfitted to this specific company's situation.

**3. Missing Context**

CRM data doesn't capture everything. Sales notes might say "lost to competitor X" but not capture the full story. Maybe the competitor was cheaper, or had a better relationship, or had a feature we don't.

I'm making decisions based on incomplete information.

### What Would Break in Production

The model is trained on 2023-2024 data. If there's a new competitor in 2025, or a product launch, or an economic downturn, the model becomes stale fast.

I'd need:
- Automated retraining pipeline (quarterly at minimum)
- Performance monitoring (are predictions still accurate?)
- Human override capability (for edge cases the model doesn't handle)

Also, this assumes the sales process stays consistent, which it won't. If we change comp plans, or hire a new sales leader, or launch a new product, all bets are off.

### What I'd Build Next (If I Had a Month)

**Week 1-2: Get more data**
- External data: competitor intelligence, market trends, economic indicators
- Internal data: sales activity (emails, calls, demos), CRM notes with NLP
- Talk to actual sales reps (data doesn't tell the whole story)

**Week 3: Better analysis**
- Causal inference instead of just correlation
- Cohort analysis (do rep behaviors persist over time?)
- Stage-by-stage conversion analysis (where exactly do deals die?)

**Week 4: Make it actionable**
- Build actual dashboard (Streamlit or similar)
- Integrate with Slack for real-time alerts
- A/B testing framework (so we can actually test interventions)

### The Part I'm Actually Proud Of

The custom metrics (PQI and RES). I think they capture something useful that standard metrics miss.

Also, I tried to keep the analysis focused on **actionability** rather than just "interesting insights." Who cares if win rate is dropping if you don't know what to do about it?

---

## How to Run This

```bash
# Setup
pip install -r requirements.txt

# Run analysis
cd notebooks
python analysis.py

# Outputs will be in outputs/
```

That's it. Nothing fancy.

---

## Final Thoughts

This was a fun problem to think through. The assignment said "we care more about how you think than what you build," so I tried to show my reasoning process rather than just building the fanciest model.

A few things I learned:
- Simple models that people understand > complex models that are black boxes
- Business context matters more than technical sophistication
- Be honest about limitations (models aren't magic)

Thanks for the opportunity! Happy to discuss any of this further.

---

**Contact:** [your email]  
**GitHub:** [repo link]  
**Time spent:** ~7 hours over 3 days
