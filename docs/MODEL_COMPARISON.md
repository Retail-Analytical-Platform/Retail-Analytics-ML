## Know what the data looks like

From the dataset page, this is a synthetic retail chain:

- 20 stores
- 2,000 loyalty customers
- 400 products in 4 categories
- 731 days (two years, 2024 and 2025)
- One row in the sales table is one sale line (one product bought by one customer in one store on one day)

Facts from the page that matter for forecasting:

- Revenue per line is very skewed: the median is about 13.78 but the mean is about 56.24, with a maximum of 4,374. A few big lines pull the average up.
- Quantity per line is small: 1 to 6, mean about 1.58.
- Revenue grows from 82,000 in January 2024 to 148,000 in December 2025, about 1.8 times.
- There is a yearly pattern: a strong peak in November and December and a drop in January (the January 2025 month is about 22 percent lower than December 2024).
- Saturdays are the busiest days, and different categories sell more in different seasons.
- Category shares are fixed: Electronics 42 percent, Home 26 percent, Apparel 20 percent, Grocery 12 percent of revenue.

---

## Work out how big the target is

You can estimate the average value of your target from your own results:

```
average target = MAE / WAPE
               = 2.1027 / 0.7133
               = about 2.95
```
---

## The scoreboard

|Model|WAPE|MAE|RMSE|sMAPE|Bias|
|---|---|---|---|---|---|
|CatBoost|0.7133|2.1027|2.5364|87.20|0.068|
|LightGBM|0.7135|2.1033|2.5371|87.19|0.070|
|XGBoost|0.7143|2.1056|2.5393|87.35|0.056|
|naive|0.9240|2.7299|3.4954|113.94|-0.598|
|seasonal_naive|0.9874|2.9129|3.6995|113.00|0.037|

Training time: LightGBM 0.49 seconds, CatBoost 1.88 seconds, XGBoost 2.69 seconds.

---

## Why the result is GOOD

### The models beat both baselines clearly

The best baseline is not seasonal naive, it is plain naive (WAPE 0.924, lower than 0.987). Against that stronger baseline:

- WAPE improves from 0.924 to about 0.714, which is about 23 percent lower error.
- MAE improves from 2.73 to 2.10, about 23 percent lower.
- RMSE improves from 3.50 to 2.54, about 27 percent lower.

Against seasonal naive the gain is the 27.7 percent in your last column. Always quote the gain against the strongest baseline, because that is the honest number.

### The gain is bigger than the noise

The WAPE std of the models is about 0.038. The gap to naive is about 0.21, which is more than five times that spread. That means the improvement is real and not a lucky fold.

### The models are not biased

Bias is about 0.06 to 0.07 on a target that averages about 3, so roughly 2 percent. You are not systematically over-forecasting or under-forecasting. Naive has a bias of -0.60, which is about 20 percent of the average, so it consistently under-forecasts.

### Three different algorithms agree

CatBoost, LightGBM and XGBoost land within 0.001 of each other on WAPE. This tells you the pipeline is healthy. If one model had been wildly better, that would suggest a bug or leakage (the model accidentally seeing the future).

### It is cheap

LightGBM trains in half a second. You can iterate quickly.

---

## Why the result is BAD (or at least limited)

### WAPE of 0.71 is high in absolute terms

On a smooth, high volume product (think bottled water in a big supermarket), a good WAPE is often 0.10 to 0.30. At 0.71 you are far from that. But that comparison is unfair for this data, see the next points.

### Your target is small and sparse

A target averaging about 3 units is a count, not a smooth curve. Think about the size of the space:

- 400 products times 20 stores times 731 days is roughly 5.8 million possible product-store-day combinations.
- The sales table only has 43,900 lines.

So at fine granularity most combinations have zero or a few sales. When values are this small, one extra customer changes the number by a large percentage. That is why sMAPE is about 87 percent and why percentage metrics look terrible even when the forecast is sensible.

### sMAPE is misleading here

sMAPE gives a huge penalty when the actual value is 0 and you predict anything above 0. On sparse retail data this inflates the score. Use WAPE and MAE as your main metrics and treat sMAPE as a secondary check.

### Part of the error can never be removed

The page says this dataset is synthetic: the trend, season and category shares are declared, and the individual sales are randomly generated around them. Randomness is, by definition, unpredictable.

A thought experiment to feel this. Suppose daily sales were a pure Poisson count (the textbook model for "random events happening at an average rate") with a true average of about 3, and suppose you knew that average perfectly. Even then your best possible MAE would be about 1.3, which is a WAPE of roughly 0.45. This is a rough illustration and not a measured limit of your data (your real target may be more or less noisy than Poisson), but it shows that with small counts, a WAPE well above zero is normal even for a perfect model.

So your 0.71 is likely not "70 percent of the problem solved wrong". A large part of it may simply be noise that no model can see.

### The models have hit a ceiling

Three different algorithms give the same score. That means the limit is not the algorithm, it is the information you gave them. Tuning hyperparameters (learning rate, tree depth) will move WAPE by maybe 0.005. New information or a different aggregation level can move it far more.

---

## Why does naive beat seasonal naive?

This confuses many beginners, so here is the reason.

- Naive copies the last value.
- Seasonal naive copies the value from one season ago.

On smooth data, seasonal naive wins when there is a strong repeating pattern. On sparse noisy data, both copy one noisy observation. Seasonal naive copies a noisy number from a week (or year) ago, and the actual value is also noisy, so you pile noise on noise. Naive is at least closer in time and picks up the recent level.

It does not mean there is no seasonality in your data. It means a single copied value is too noisy to carry the seasonal signal. The ML models can average over many rows, so they extract that signal properly. That is exactly where they beat both baselines.

Also notice naive has a large fold-to-fold std (0.106 versus 0.038 for the models), so it is unstable, which is another reason not to trust it.

---

## Final verdict

|Question|Answer|
|---|---|
|Is the ML better than the baselines?|Yes, clearly. About 23 percent lower error than the best baseline.|
|Is the pipeline trustworthy?|Yes. Stable folds, near zero bias, three models agree.|
|Is the absolute accuracy high?|No. WAPE 0.71 is high.|
|Is that a failure?|Probably not. Small sparse counts plus synthetic randomness make a high WAPE expected.|
|Will tuning help?|Very little. Change the data or the level of detail instead.|

---

## What to try next

1. Check how many target values are zero. Print the percentage of zeros and the distribution of the target. This confirms the sparsity story.
2. Aggregate. Forecast weekly instead of daily, or per category per store instead of per product per store. Error percentages drop sharply as numbers get larger. Compare WAPE at each level.
3. Break WAPE down by segment: by category, by store, by high volume versus low volume products. You will likely find a good WAPE on big series and a terrible one on the long tail of tiny series.
4. Add features that match what the page says drives the data: day of week (Saturday effect), month and week of year (November and December peak), a trend feature, category, store, lag features, and rolling averages.
5. Try a count-friendly loss function. Poisson or Tweedie objectives exist in LightGBM, XGBoost and CatBoost, and they suit small non-negative counts better than squared error.
6. Check leakage once. Confirm every feature for a given date uses only information from before that date, and that validation folds are strictly later in time than training folds.
7. Report the result honestly: "28 percent better than seasonal naive and 23 percent better than naive, WAPE 0.71, limited by sparse small counts".
