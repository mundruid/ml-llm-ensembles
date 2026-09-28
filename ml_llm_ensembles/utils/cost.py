"""Dollar-cost arithmetic for the inference cost benchmark (experiment 14).

Two regimes, kept separate because they are not the same kind of number: a
hosted API is metered per token, a local model is amortized hardware time.
Both are expressed as USD per one million classified records so that the
Table 6 rows are comparable.
"""

# Published list prices, USD per million tokens, as (input, output).
API_PRICES: dict[str, tuple[float, float]] = {
    "claude-sonnet-4-6": (3.0, 15.0),
}

# USD per GPU-hour used to amortize locally served models. This is an assumption,
# not a measurement: it is the rental rate of a comparable cloud GPU, and it must
# be reported alongside any figure derived from it.
DEFAULT_GPU_HOURLY_USD = 0.40

# USD per CPU-hour, for the rows Table 6 places on CPU. Same caveat as above.
DEFAULT_CPU_HOURLY_USD = 0.05


def api_dollars_per_million(
    mean_input_tokens: float,
    mean_output_tokens: float,
    price_in: float,
    price_out: float,
) -> float:
    """Metered cost of classifying 1e6 records.

    price_in/price_out are USD per 1e6 tokens, so the 1e6 records and the 1e6
    tokens cancel and the mean per-record token counts scale the prices directly.
    """
    return mean_input_tokens * price_in + mean_output_tokens * price_out


def local_dollars_per_million(throughput_rps: float, hourly_rate: float) -> float:
    """Amortized cost of classifying 1e6 records on hardware rented at hourly_rate."""
    return (1e6 / throughput_rps) / 3600.0 * hourly_rate
