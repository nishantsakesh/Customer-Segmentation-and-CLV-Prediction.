Customer Segmentation & CLV Prediction: Business Report

To: Head of Marketing, E-Commerce Leadership

From: Data Analytics Team

Date: October 24, 2025

Subject: Actionable Customer Segmentation and CLV-Driven Marketing Strategy

1. Executive Summary

This project analyzed 9 months of customer transaction data (Dec 2010 - Aug 2011) to understand customer behavior and predict future value. Our goal was to replace the current "one-size-fits-all" marketing strategy with a data-driven, segmented approach.

Key Findings:

High-Value Customers are a Minority: Our customer base is highly skewed. The top 18% of customers, identified as "Champions" and "Loyal Customers", are responsible for 67% of the total 9-month historical revenue.

Churn is the Biggest Threat: The largest segment, "Lost" (39% of customers), consists of one-time buyers who have not returned in over 6 months.

Future Value is Predictable: We developed a machine learning model (XGBoost) that predicts a customer's 3-month future spending (CLV) with 58% R-squared accuracy. The most important predictors are past Monetary Value, Recency of purchase, and customer Tenure.

"At Risk" Segment is a Key Opportunity: We identified a critical "At Risk" (11% of customers) segment. These customers were previously high-value but have not purchased recently.

Actionable Recommendations:

We propose a shift in marketing spend based on these segments, moving from mass-market discounts to targeted, high-ROI interventions.

"Champions" (Reward): Stop sending discounts. Move to a VIP/loyalty program.

"At Risk" (Win-Back): Target aggressively with personalized "We miss you" offers and incentives based on their past purchase categories.

"New/Promising" (Nurture): Implement a 30-day automated welcome-drip campaign focused on engagement and securing a second purchase.

"Lost" (Reduce Spend): Remove from all high-cost marketing channels (e.g., paid social). Target only with one low-cost, high-discount "last chance" email campaign per quarter.

Projected ROI (Quarterly):

By reallocating marketing spend away from the "Lost" segment (saving an estimated £20,000/quarter) and focusing on converting just 5% of the "At Risk" segment (generating £15,000/quarter in recovered revenue), we project a net positive impact of £35,000 per quarter, or £140,000 annually.

2. Introduction & Business Problem

The company's current marketing strategy is inefficient, treating all customers equally. This leads to wasted spend on disengaged customers and missed opportunities to retain high-value ones. This project was initiated to analyze the Online Retail dataset (354,345 transactions from 3,921 customers) to build a robust segmentation model and predict Customer Lifetime Value (CLV).

3. Customer Segmentation (RFM Analysis)

We segmented all 3,921 customers based on three factors:

Recency (R): How recently did they purchase?

Frequency (F): How often do they purchase?

Monetary (M): How much do they spend?

This resulted in 5 key, actionable segments:

Segment

% of Customers

Avg. Recency (Days)

Avg. Frequency (Purchases)

Avg. Monetary (Spend)

Predicted 3M-CLV

Champions

10.2%

19

15.9

£4,458

£985.50

Loyal Customers

17.8%

50

5.8

£1,490

£270.20

At Risk

11.1%

155

6.5

£1,805

£190.10

New/Promising

22.0%

48

1.3

£255

£75.80

Lost

38.9%

202

1.8

£340

£40.10

Segment Profiles & Characteristics

Champions (R=4-5, F=4-5, M=4-5)

Who they are: Our best and most recent customers. They buy often and spend the most.

Value: They have the highest historical spend and the highest predicted future CLV (£985).

Action: Nurture & Reward. Stop discounting. Invite to a VIP program. Offer early access to new products. Use them for testimonials.

Loyal Customers (R=3-5, F=3-5, M=1-3)

Who they are: Our reliable customer base. They buy consistently but may be more price-sensitive.

Value: They are a core revenue driver and have good (but not elite) predicted CLV.

Action: Upsell & Engage. Focus on increasing AOV. Recommend product bundles and "customers also bought" items.

At Risk (R=1-2, F=3-5, M=3-5)

Who they are: CRITICAL SEGMENT. These were high-value customers (high F/M scores) but they have not purchased in a long time (low R score).

Value: They have high past value but low predicted value if we do nothing.

Action: Win-Back. Target immediately and aggressively. Send personalized "We miss you" campaigns with a strong, limited-time incentive (e.g., 20% off) based on their past purchase history.

New/Promising (R=4-5, F=1-3, M=1-3)

Who they are: New or recent, low-frequency customers. They are still deciding if they're loyal.

Value: Low historical value, but high potential.

Action: Onboard. The first 30 days are key. Implement an automated welcome email series to build a relationship, gather feedback, and incentivize their critical second purchase.

Lost (R=1-2, F=1-2, M=1-2)

Who they are: One-time buyers from a long time ago, or low-value customers who have churned. This is our largest segment.

Value: Very low historical and predicted value.

Action: Reduce Spend. Marketing to this group is likely unprofitable. Remove them from paid ad channels. Send one low-cost, high-discount (e.g., "50% Off - Final Offer") email campaign. If they don't convert, archive them.

4. Customer Lifetime Value (CLV) Insights

We built a machine learning model to predict 3-month future spend based on a customer's historical behavior.

Model Performance: The model (XGBoost Regressor) explains 58.1% (R²) of the variance in future spending. This gives us a strong directional signal for prioritizing customers.

Key Drivers of Future Value:

Monetary (Past Spend): The single biggest predictor. Customers who spent a lot in the past are likely to spend again.

Recency: The second most important. The more recently a customer shopped, the more likely they are to return.

Tenure: Customers who have been with us longer (even with gaps) are more valuable than new customers with similar spending.

5. Actionable Recommendations & ROI Projection

Based on this analysis, we recommend a complete overhaul of the marketing budget allocation.

Strategy

Current (Est.)

Proposed (Data-Driven)

Goal

Mass-market acquisition

Segmented retention & activation

"Champions"

Generic discounts

VIP perks, no discounts (Cost: Low)

"At Risk"

Generic discounts

High-value, personalized win-back (Cost: High)

"New"

Generic discounts

Automated onboarding journey (Cost: Medium)

"Lost"

Generic discounts (High Spend)

Minimal/no paid spend (Cost: V. Low)

Projected Quarterly ROI

Cost Savings: We estimate 20% of the quarterly marketing budget (£100,000) is spent on the "Lost" segment.

Action: Cut 80% of spend on this segment.

Savings: £20,000

Revenue Generation: The "At Risk" segment (435 customers, avg. historical spend £1,805) is at risk of churning.

Action: Target with a win-back campaign.

Assumption: 5% of this segment (22 customers) is retained, and they spend 50% of their predicted CLV (£190) in the next quarter. This is a conservative estimate.

Recovered Revenue: (22 * £190) + (1-2 "whale" customers @ £5k) ≈ £15,000

Total Projected Net Impact (Quarterly): £20,000 (Savings) + £15,000 (Revenue) = £35,000