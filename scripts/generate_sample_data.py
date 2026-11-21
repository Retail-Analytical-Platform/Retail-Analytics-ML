import argparse
from pathlib import Path

import numpy as np
import pandas as pd

DEFAULT_OUT = Path(__file__).resolve().parents[1] / "src" / "tests" / "fixtures"
START, END = "2024-01-01", "2025-12-31"
CATEGORIES = ["Grocery", "Electronics", "Apparel", "Home"]
REGIONS = ["North", "South", "East"]
STORE_TYPES = ["Mall", "Street", "Mall"]


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--out", type=Path, default=DEFAULT_OUT)
    p.add_argument("--stores", type=int, default=3)
    p.add_argument("--products-per-category", type=int, default=2)
    p.add_argument("--customers", type=int, default=20)
    p.add_argument("--keep", type=float, default=0.5, help="share of the daily grid kept as sales")
    p.add_argument("--seed", type=int, default=42)
    a = p.parse_args()

    rng = np.random.default_rng(a.seed)
    a.out.mkdir(parents=True, exist_ok=True)

    days = pd.date_range(START, END, freq="D")
    dim_date = pd.DataFrame({
        "date_key": [int(d.strftime("%Y%m%d")) for d in days],
        "full_date": days,
        "day_of_week": days.day_name(),
        "month_name": days.month_name(),
        "quarter": [f"Q{q}" for q in days.quarter],
        "is_weekend": days.day_of_week >= 5,
    })

    keys = range(1, a.customers + 1)
    dim_customer = pd.DataFrame({
        "customer_key": keys,
        "customer_name": [f"Customer {i:03d}" for i in keys],
        "email": [f"customer{i:03d}@example.com" for i in keys],
        "city": rng.choice(["Yogyakarta", "Edinburgh", "London"], a.customers),
        "country": "Exampleland",
        "segment": rng.choice(["Consumer", "Corporate", "Home Office"], a.customers),
    })

    n_sub = a.products_per_category
    products = []
    for i, category in enumerate(CATEGORIES):
        for j in range(n_sub):
            sub = ("Basic", "Premium")[j] if n_sub == 2 else f"Line {j + 1:03d}"
            price = 10.0 * (i + 1) * (1 + j % 5)
            products.append({
                "product_key": i * n_sub + j + 1,
                "product_name": f"{category} {sub}",
                "category": category,
                "subcategory": sub,
                "unit_cost": round(price * 0.6, 2),
                "list_price": price,
            })
    dim_product = pd.DataFrame(products)

    store_keys = list(range(1, a.stores + 1))
    dim_store = pd.DataFrame({
        "store_key": store_keys,
        "store_name": ["Store A", "Store B", "Store C"] if a.stores == 3
                      else [f"Store {k:02d}" for k in store_keys],
        "region": [REGIONS[k % 3] for k in range(a.stores)],
        "store_type": [STORE_TYPES[k % 3] for k in range(a.stores)],
    })

    grid = pd.MultiIndex.from_product(
        [days, dim_store["store_key"], dim_product["product_key"]],
        names=["date", "store_key", "product_key"],
    ).to_frame(index=False)
    grid = grid[rng.random(len(grid)) < a.keep].reset_index(drop=True)
    n = len(grid)

    product = dim_product.set_index("product_key").loc[grid["product_key"]]
    price = product["list_price"].to_numpy()
    quantity = rng.integers(1, 6, n)
    discount = rng.choice([0.0, 0.05, 0.10, 0.20], n, p=[0.6, 0.2, 0.15, 0.05])
    customer = pd.array(rng.integers(1, a.customers + 1, n), dtype="Int64")
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
        df.to_csv(a.out / f"{name}.csv", index=False)
        print(f"{name}: {len(df)} rows")


if __name__ == "__main__":
    main()