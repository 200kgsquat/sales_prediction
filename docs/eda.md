Exploratory Data Analysis (EDA) Summary
1. Monthly Sales Distribution
Key Insight:

Strong seasonal fluctuations with consistent peaks during New Year/holiday periods

General downward trend in sales over time

Highest sales observed in November/December (holiday season)

Lowest sales during summer months (June-August)

2. Category Performance
Top Categories (Sales >10,000 units):

PC Games 

Mac Games 

PS3 Games 

XBOX 360 Games 

Mobile Games 

Payment Cards 

Movies/Music/Games 

Gifts 

Blank Media 

Premium Categories:

Consoles (highest average price)

PS4/XBOX accessories

Collectible cards

3. Product Popularity
Top 10 Products:

Radio-controlled flying platform (PC Games)

Gift voucher 300 rub (Gifts)

Delivery of goods (Payment Cards)

PS3 game accessories

Payment cards for online games

Sales Concentration:

Top products account for disproportionately high sales volume

Gaming-related items dominate bestsellers

4. Shop Performance
Top Performing Shops:

Интернет-магазин ЧС (Online store CHS) - exceptional outlier

Moscow stores (Ярославль, Москва ТЦ "Семеновский")

St. Petersburg locations

Underperforming Shops:

Outlying regional stores

Shops with limited category assortment

5. Time Series Analysis
Key Patterns:

Clear downward trend (2013-2015)

Strong yearly seasonality (period=12 months)

Autocorrelation peaks at 12-month intervals

Moving Average:

3-month MA confirms seasonal patterns

Smoothed trend shows accelerating decline in late 2015

6. Price-Sales Relationship
Negative Correlation (r = -0.11):

Higher-priced items → Lower sales volume

Essential/low-cost goods drive volume

Outlier Behavior:

Most transactions concentrated in <10,000 RUB range

7. Category-Shop Interactions
Specialization Patterns:

Online store CHS dominates category 9 (Blank Media)

Physical stores specialize in location-specific assortments

8. Seasonal Patterns by Category
Category-Specific Seasonality:

Movies/Music/Games : Strong Dec/Jan peaks

Payment Cards : Consistent quarterly spikes

PC Games : Holiday-driven spikes

Gifts : Year-end dominance

Universal Patterns:

December peak across all categories

Summer trough (June-August)

9. Key Insights for Modeling
Time Features:

Month/seasonal indicators critical

Lag features (1, 2, 3, 12 months)

Holiday period markers

Geographical Features:

City-level indicators (Moscow/St.Petersburg/regional)

Shop performance tiers

Category Features:

Presence of top categories in shop

Category price tier (premium/essential)

Category-season interaction terms

Special Features:

Online vs physical store indicator

Category exclusivity markers

Shop-category performance history

Conclusion: Sales patterns show strong time dependency, geographical influence, and category-driven behaviors. Successful models must incorporate:

Hierarchical structure (shop → category → item)

Time-series characteristics (seasonality, lags)

Store-type differentiation (online vs physical)