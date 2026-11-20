#Forecast Strategy.

##decision, rationale, fallback rule

- Target: `quantity` (units). Revenue is quantity × price × discount, so it adds
  price-mix noise unrelated to demand.
- Grain: store × day. 1 series per daily revenue, 1 series per daily quantity, 4 series by daily quantity bg category, 3 series by daily quantity by store. 1.6% on daily quantity by category and 0.4% daily quantity by store that known as zero-sales days. 
- Horizon 28 days (a multiple of 7, so the smallest allowed lag stays on the
  same weekday). Training window 180 days, provisional until the backtest (#24, #33).
- Fallback: category × day if store-day zero-day share is above ~10% on real data.

--current synthetic data report. 