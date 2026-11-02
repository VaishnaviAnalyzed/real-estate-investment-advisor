# EDA Findings (auto-generated)

Data set: 250,000 cleaned listings.


## Price & Size

### Q1. What is the distribution of property prices?
Prices range from ₹10L to ₹500L (mean ₹255L, median ₹254L, skew 0.01). The distribution is almost flat, i.e. close to uniform rather than the right-skewed shape of real markets.

_Figure: `reports/figures/q1.html`_

### Q2. What is the distribution of property sizes?
Sizes span 500 - 5,000 sq ft (median 2,747); the spread is again nearly uniform.

_Figure: `reports/figures/q2.html`_

### Q3. How does price per sq ft vary by property type?
Mean price per sq ft by type. Highest: **Independent House** (13,102.05 ₹/sqft); lowest: **Villa** (13,025.61 ₹/sqft) - a gap of only 0.6%.

_Figure: `reports/figures/q3.html`_

### Q4. Is there a relationship between property size and price?
Correlation between size and total price is r = -0.003: practically **no linear relationship**. In this data set price does not scale with area, so price per sq ft is driven mostly by size itself (r = -0.61).

_Figure: `reports/figures/q4.html`_

### Q5. Are there outliers in price per sq ft or size?
IQR rule flags 19,723 price-per-sq-ft outliers (7.9%), 0 price outliers and 0 size outliers. Outliers appear only in price/sq ft (small flats with high totals). They are **flagged, not removed**, as they are valid listings.

_Figure: `reports/figures/q5.html`_


## Location

### Q6. Average price per sq ft by state
Highest: **Karnataka** (13,252.20 ₹/sqft); lowest: **Punjab** (12,931.49 ₹/sqft) - a gap of only 2.5%.

_Figure: `reports/figures/q6.html`_

### Q7. Average property price by city
Highest: **Bangalore** (258.46L); lowest: **Cuttack** (250.79L) - a gap of only 3.1%.

_Figure: `reports/figures/q7.html`_

### Q8. Median age of properties by locality
Across 500 localities the median age ranges 16 - 21 years (overall median 18). Oldest: Locality_72, Locality_403, Locality_90.

_Figure: `reports/figures/q8.html`_

### Q9. How is BHK distributed across cities?
Every city has a near-identical BHK split (each BHK 1-5 holds roughly 20% of listings, city-to-city spread ±0.5 pts).

_Figure: `reports/figures/q9.html`_

### Q10. Price trends - top 5 most expensive localities
Top 5 localities by mean price: Locality_395 (₹275L), Locality_366 (₹273L), Locality_497 (₹270L), Locality_360 (₹270L), Locality_203 (₹269L). The dataset has no sale date, so the trend is shown by year built; lines are noisy with no consistent upward trend.

_Figure: `reports/figures/q10.html`_


## Relationships

### Q11. How are numeric features correlated?
Strongest pair: **Size_in_SqFt ↔ Price_per_SqFt_INR** (r = -0.61); this is mechanical, because price/sq ft = price ÷ size. Apart from those derived pairs almost all correlations are ≈ 0.

_Figure: `reports/figures/q11.html`_

### Q12. Nearby schools vs price per sq ft
Correlation with price per sq ft: r = -0.000. Average ₹/sq ft ranges only 12,888 - 13,292 across the scale.

_Figure: `reports/figures/q12.html`_

### Q13. Nearby hospitals vs price per sq ft
Correlation with price per sq ft: r = -0.000. Average ₹/sq ft ranges only 12,917 - 13,257 across the scale.

_Figure: `reports/figures/q13.html`_

### Q14. Price by furnished status
Mean price by furnishing level. Highest: **Unfurnished** (254.98L); lowest: **Semi-furnished** (254.33L) - a gap of only 0.3%.

_Figure: `reports/figures/q14.html`_

### Q15. Price per sq ft by facing direction
Highest: **West** (13,140.02 ₹/sqft); lowest: **East** (13,023.70 ₹/sqft) - a gap of only 0.9%.

_Figure: `reports/figures/q15.html`_


## Investment & Amenities

### Q16. Properties per owner type
Listings: Broker 83,479 (33.4%), Owner 83,268 (33.3%), Builder 83,253 (33.3%).

_Figure: `reports/figures/q16.html`_

### Q17. Properties per availability status
Listings: Under_Construction 125,035 (50.0%), Ready_to_Move 124,965 (50.0%). (No 'Sold' status exists in this data.)

_Figure: `reports/figures/q17.html`_

### Q18. Does parking space affect price?
Properties with parking differ from those without by +0.1% in total price and -0.2% in price per sq ft - negligible.

_Figure: `reports/figures/q18.html`_

### Q19. How do amenities affect price per sq ft?
Effect of each individual amenity on mean ₹/sq ft: Clubhouse +0.2%, Garden +0.1%, Gym +0.8%, Playground -0.2%, Pool +0.3%.

_Figure: `reports/figures/q19.html`_

### Q20. Public transport vs price/sq ft and investment potential
Price/sq ft barely changes with transport access, but the modelled 5-year return rises from 54.4% (Low) to 61.6% (High) because transport is a growth driver in the future-price rules.

_Figure: `reports/figures/q20.html`_


## Step-2 analyses

### S1. Price trends by city
Price by city (year-built trend). Highest: **Bangalore** (258.46L); lowest: **Cuttack** (250.79L) - a gap of only 3.1%.

_Figure: `reports/figures/s1.html`_

### S2. Area vs investment return
Correlation of size with expected 5-yr return: r = 0.001 (return depends on growth rules, not area). However, the Good-Investment rate climbs from 66% to 39% with size because bigger homes have a lower price per sq ft, which the label rewards.

_Figure: `reports/figures/s2.html`_

### S3. Safety (security) vs Good-Investment classification
**Note:** the data set has no crime-rate column, so the 'Security' flag is used as a safety proxy. Good-investment rate: Security=No: 46.4%, Security=Yes: 60.2%.

_Figure: `reports/figures/s3.html`_

### S4. Infrastructure score vs resale value
Correlation between infrastructure score and projected resale price: r = 0.012. Better infrastructure raises expected growth only through the transport factor.

_Figure: `reports/figures/s4.html`_
