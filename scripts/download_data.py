"""Download prices (Yahoo) and factors (FRED) fresh and cache to data/*.parquet.

Usage: python -m scripts.download_data
"""
from shocklab.data import load_panel


def main() -> None:
    p = load_panel(refresh=True)
    print(f"prices {p.prices.shape}, windows {p.Y.shape}, {p.prices.index[0].date()} -> {p.prices.index[-1].date()}")


if __name__ == "__main__":
    main()
