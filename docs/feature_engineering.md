1. Temporal Features
season:

Why? EDA revealed strong seasonal patterns (e.g., holiday spikes, summer slumps)

Impact: Helps model understand cyclical demand fluctuations

month_sin/month_cos:

Why? Solves the "December-January discontinuity" problem

Impact: Preserves cyclical relationships between months (e.g., Dec ↔ Jan are adjacent in vector space)

2. Lag Features (1,2,3,12 months)
item_cnt_month_lag_[1,2,3]:

Why? EDA autocorrelation showed strong short-term dependencies

Impact: Captures momentum effects (e.g., popular items stay popular)

item_cnt_month_lag_12:

Why? Seasonal decomposition revealed yearly cycles

Impact: Models annual recurring patterns (e.g., holiday gifts)

3. Rolling Statistics
rolling_median_3/6:

Why? Smoothes short-term noise while preserving trends

Impact: Reduces overreaction to temporary fluctuations (e.g., promotions)

4. Aggregated Features
item_avg_sales_month_lag1:

Why? EDA showed certain items consistently outperform

Impact: Flags inherently popular products

shop_item_avg_lag1:

Why? Store-item combinations have unique sales profiles

Impact: Captures location-specific demand (e.g., regional preferences)

5. Price Dynamics
price_change/price_increasing:

Why? EDA found negative price-sales correlation

Impact: Models price elasticity effects

price_range:

Why? Premium vs. budget items have different demand curves

Impact: Segments products by price tier

6. Shop Metadata
city:

Why? Moscow, St.Petersburg etc. shops showed higher sales

Impact: Encodes regional purchasing power

type_of_shop:

Why? Online vs physical stores had divergent sales patterns

Impact: Differentiates sales channels

7. Logarithmic Transform
log_item_cnt_month_lag_1:

Why? Data has outliers

Impact: Normalizes target variable for linear models

Strategic Rationale
Addressing Observed Patterns:

Seasonality → Temporal features

Regional variance → Shop metadata

Price sensitivity → Price dynamics

Handling Data Challenges:

Short-term noise → Rolling medians

Yearly cycles → 12-month lag

Skewed targets → Log transform

Business Realities:

Product lifecycles → Item-level aggregates

Location effects → City segmentation

Promotional impact → Short-term lags

Test Data Robustness:

Conservative zero-filling for new items

Price inheritance from training data

Time-aligned feature calculation

Evidence-Based Selection

                  item_price    
   item_avg_sales_month_lag1    
        item_cnt_month_lag_1    
              date_block_num    
                price_change    
            item_price_lag_1    
          shop_item_avg_lag1    
            rolling_median_3    
            item_category_id   
    log_item_cnt_month_lag_1   
       item_cnt_month_lag_2   
           rolling_median_6   
           price_increasing   
                  month_cos   