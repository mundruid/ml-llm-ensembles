#!/usr/bin/env python3
"""Hand-computed checks for the cost arithmetic behind experiment 14.

    uv run python experiments/tests_14_cost.py

Plain asserts; exit code non-zero on the first failure.
"""
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_ROOT))

from ml_llm_ensembles.utils.cost import (
    API_PRICES, DEFAULT_GPU_HOURLY_USD,
    api_dollars_per_million, local_dollars_per_million,
)

# 1. Metered regime. A 1000-token prompt and a 10-token verdict at Sonnet list
# price: 1000 * $3/1e6 tokens * 1e6 records = $3000, plus 10 * $15 = $150.
price_in, price_out = API_PRICES["claude-sonnet-4-6"]
assert (price_in, price_out) == (3.0, 15.0)
assert api_dollars_per_million(1000, 10, price_in, price_out) == 3150.0

# 2. Amortized regime. 1e6/3600 records per second clears a million records in
# exactly one hour, so the bill is exactly one hour of rent.
assert abs(local_dollars_per_million(1e6 / 3600, DEFAULT_GPU_HOURLY_USD)
           - DEFAULT_GPU_HOURLY_USD) < 1e-12

# 3. Twice the throughput, half the cost.
assert abs(local_dollars_per_million(500, 0.40)
           - 2 * local_dollars_per_million(1000, 0.40)) < 1e-12

# 4. The two regimes are orders of magnitude apart, which is the point of Table 6.
assert local_dollars_per_million(50, DEFAULT_GPU_HOURLY_USD) < 0.01 * \
    api_dollars_per_million(1000, 10, price_in, price_out)

print("all cost checks passed")
