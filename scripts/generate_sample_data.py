from pathlib import Path

import numpy as np
import pandas as pd

OUT = Path(__file__).resolve().parents[1] / "src" / "tests" / "fixtures"
START, END = "2024-01-01", "2025-12-31"
SEED = 42
N_CUSTOMERS = 20
CATEGORIES = ["Grocery", "Electronics", "Apparel", "Home"]


def main() -> None:
    rng = np.random.default_rng(SEED)
    OUT.mkdir(parents=True, exist_ok=True)

    days = pd.date_range(START, END, freq="D")
    dim_date = pd.DataFrame({
        "date_key": [int(d.strftime("%Y%m%d")) for d in days],
        "full_date": days,
        "day_of_week": days.day_name(),
        "month_name": days.month_name(),
        "quarter": [f"Q{q}" for q in days.quarter],
        "is_weekend": days.day_of_week >= 5,
    })

    keys = range(1, N_CUSTOMERS + 1)
    dim_customer = pd.DataFrame({
        "customer_key": keys,
        "customer_name": [f"Customer {i:03d}" for i in keys],
        "email": [f"customer{i:03d}@example.com" for i in keys],
        "city": rng.choice(["Yogyakarta", "Edinburgh", "London"], N_CUSTOMERS),
        "country": "Exampleland",
        "segment": rng.choice(["Consumer", "Corporate", "Home Office"], N_CUSTOMERS),
    })

    products = []
    for i, category in enumerate(CATEGORIES):
        for j, subcategory in enumerate(("Basic", "Premium")):
            price = 10.0 * (i + 1) * (j + 1)
            products.append({
                "product_key": i * 2 + j + 1,
                "product_name": f"{category} {subcategory}",
                "category": category,
                "subcategory": subcategory,
                "unit_cost": round(price * 0.6, 2),
                "list_price": price,
            })
    dim_product = pd.DataFrame(products)

    dim_store = pd.DataFrame({
        "store_key": [1, 2, 3],
        "store_name": ["Store A", "Store B", "Store C"],
        "region": ["North", "South", "East"],
        "store_type": ["Mall", "Street", "Mall"],
    })

    # One candidate sale per day, store and product; keep about half, so some
    # series have zero-sales days (#19 has to handle those).
    grid = pd.MultiIndex.from_product(
        [days, dim_store["store_key"], dim_product["product_key"]],
        names=["date", "store_key", "product_key"],
    ).to_frame(index=False)
    grid = grid[rng.random(len(grid)) < 0.5].reset_index(drop=True)
    n = len(grid)

    product = dim_product.set_index("product_key").loc[grid["product_key"]]
    price = product["list_price"].to_numpy()
    quantity = rng.integers(1, 6, n)
    discount = rng.choice([0.0, 0.05, 0.10, 0.20], n, p=[0.6, 0.2, 0.15, 0.05])
    customer = pd.array(rng.integers(1, N_CUSTOMERS + 1, n), dtype="Int64")
    customer[rng.random(n) < 0.05] = pd.NA  # guest sales
    seconds = rng.integers(8 * 3600, 21 * 3600, n)

    fact_sales = pd.DataFrame({
        "sale_id": np.arange(1, n + 1),
        "date_key": grid["date"].dt.strftime("%Y%m%d").astype(int),
        "product_key": grid["product_key"],
        "customer_key": customer,
        "store_key": grid["store_key"],
        "sold_at": grid["date"] + pd.to_timedelta(seconds, unit="s"),
        "quantity": quantity,
        "unit_price": price,
        "discount_pct": discount,
        "revenue": np.round(quantity * price * (1 - discount), 2),
        "category": product["category"].to_numpy(),
    })

    tables = {
        "dim_customer": dim_customer,
        "dim_date": dim_date,
        "dim_product": dim_product,
        "dim_store": dim_store,
        "fact_sales": fact_sales,
    }
    for name, df in tables.items():
        df.to_csv(OUT / f"{name}.csv", index=False)
        print(f"{name}: {len(df)} rows")


if __name__ == "__main__":
    main()