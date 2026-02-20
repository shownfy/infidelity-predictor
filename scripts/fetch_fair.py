"""Fetch Fair's Extramarital Affairs Dataset from statsmodels."""
import os
import pandas as pd
from statsmodels.datasets import fair

OUTPUT_PATH = os.path.join(os.path.dirname(__file__), "..", "seeds", "fair_affairs_raw.csv")


def main():
    print("Fetching Fair's Affairs dataset from statsmodels...")
    data = fair.load_pandas()
    df = data.data
    print(f"  Rows: {len(df)}, Columns: {list(df.columns)}")
    df.to_csv(OUTPUT_PATH, index=False)
    print(f"  Saved to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
