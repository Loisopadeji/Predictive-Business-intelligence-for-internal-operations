# Data Dictionary — Synthetic Project Records

**File:** `Predictive_BI_for_internal_operation_data.csv`
**Rows:** 1,897 project records
**Period:** January 2021 – December 2025 (60 months)
**Status:** SYNTHETIC — generated to prototype the models below. Structure mirrors what should be pulled from real internal project/finance records before final deployment.

| Column | Type | Description |
|---|---|---|
| project_id | string | Unique project identifier (P0001...) |
| client_id | string | Unique client identifier |
| client_name | string | Synthetic client name |
| sector | category | One of the firm's 7 sectors |
| service_line | category | Strategic Planning, Market & Social Research, Data Visualization & Analytics, Political Risk Management, Stakeholder Engagement, Digital Transformation |
| start_date | date | Project start date |
| planned_end_date | date | Contracted end date |
| actual_end_date | date | Actual completion date (blank if Ongoing/Cancelled) |
| status | category | Completed, Ongoing, Cancelled |
| complexity | category | Low, Medium, High — proxy for project difficulty |
| team_size | int | Total consultants assigned |
| senior_staff_count | int | Senior consultants on the team |
| junior_staff_count | int | Junior/analyst staff on the team |
| contract_value_usd | float | Contracted project value |
| actual_cost_usd | float | Actual delivery cost (blank if Ongoing) |
| delay_days | int | Days beyond planned end date (blank if not yet completed) |
| cost_overrun_pct | float | Cost overrun as % of contract value (blank if not yet completed) |
| client_satisfaction_score | float | 1–5 post-project score (blank if not yet completed) |
| renewed_within_12m | category | Yes / No / Too Recent (can't be known yet) / N/A (cancelled) |
| acquisition_channel | category | Repeat Client, Referral, Competitive Tender, Direct Outreach |

## What each of the 5 models pulls from this table

1. **Revenue & Pipeline Forecasting** → `start_date`, `contract_value_usd`, `sector` (aggregate to monthly revenue by sector, forecast forward)
2. **Staffing & Capacity Forecasting** → `start_date`, `team_size`, `sector` (aggregate to monthly team-months by sector)
3. **Project Delivery Risk Prediction** → `complexity`, `team_size`, `sector`, `service_line` as features → predict `delay_days` / overrun (target derived from delay_days > 0)
4. **Sector Demand Trend Analysis** → `start_date`, `sector` (monthly project counts and revenue, trend + seasonality)
5. **Client Renewal Likelihood** → `client_satisfaction_score`, `delay_days`, `cost_overrun_pct`, `acquisition_channel`, `sector` as features → predict `renewed_within_12m` (only rows where this isn't "Too Recent")

## Before real data replaces this

To swap in actual company data, you need the same 19 fields pulled from: project management records (dates, team, status), finance/invoicing (contract value, actual cost), and any post-project client survey (satisfaction score). If satisfaction scores don't exist internally, that field can be dropped and Model 5 rebuilt on delay/cost signals alone.
