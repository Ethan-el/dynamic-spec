"""Pure-Python helpers for batch-size based speculative decoding."""

DynamicSpeculativeSchedule = list[tuple[int, int, int]]


def validate_and_normalize_dynamic_speculative_schedule(
    schedule: object,
) -> DynamicSpeculativeSchedule:
    """Validate inclusive ``(start_batch, end_batch, K)`` ranges."""
    if not isinstance(schedule, list) or not schedule:
        raise ValueError(
            "num_speculative_tokens_per_batch_size must be a non-empty list"
        )

    normalized: DynamicSpeculativeSchedule = []
    for entry in schedule:
        if not isinstance(entry, (list, tuple)) or len(entry) != 3:
            raise ValueError(
                "each dynamic speculative entry must be "
                "(start_batch, end_batch, num_speculative_tokens)"
            )
        start_batch, end_batch, num_speculative_tokens = map(int, entry)
        if start_batch <= 0 or end_batch <= 0:
            raise ValueError("dynamic speculative batch-size ranges must be positive")
        if start_batch > end_batch:
            raise ValueError("dynamic speculative range start must be <= end")
        if num_speculative_tokens < 0:
            raise ValueError("dynamic speculative K must be >= 0")
        normalized.append((start_batch, end_batch, num_speculative_tokens))

    normalized.sort(key=lambda entry: entry[0])
    previous_end = 0
    for start_batch, end_batch, _ in normalized:
        if start_batch <= previous_end:
            raise ValueError("dynamic speculative batch-size ranges must not overlap")
        previous_end = end_batch
    if normalized[0][0] != 1:
        raise ValueError("the first dynamic speculative range must start at batch size 1")
    return normalized


def build_dynamic_speculative_lookup(
    schedule: object,
    max_batch_size: int,
    max_num_speculative_tokens: int,
) -> list[int]:
    """Expand ranges into a direct one-indexed ``batch_size -> K`` table."""
    if max_batch_size <= 0:
        raise ValueError("max_batch_size must be > 0")
    if max_num_speculative_tokens <= 0:
        raise ValueError("max_num_speculative_tokens must be > 0")

    normalized = validate_and_normalize_dynamic_speculative_schedule(schedule)
    lookup = [0] * (max_batch_size + 1)
    next_batch_size = 1
    previous_k: int | None = None

    for start_batch, end_batch, configured_k in normalized:
        if previous_k is not None:
            for batch_size in range(
                next_batch_size,
                min(start_batch, max_batch_size + 1),
            ):
                lookup[batch_size] = min(previous_k, max_num_speculative_tokens)

        for batch_size in range(
            max(start_batch, next_batch_size),
            min(end_batch, max_batch_size) + 1,
        ):
            lookup[batch_size] = min(
                configured_k,
                max_num_speculative_tokens,
            )

        next_batch_size = max(next_batch_size, end_batch + 1)
        previous_k = configured_k
        if next_batch_size > max_batch_size:
            break

    assert previous_k is not None
    for batch_size in range(next_batch_size, max_batch_size + 1):
        lookup[batch_size] = min(previous_k, max_num_speculative_tokens)
    return lookup
