import importlib.util
import unittest
from pathlib import Path


MODULE_PATH = (
    Path(__file__).resolve().parents[1]
    / "nanovllm"
    / "dynamic_speculative.py"
)
SPEC = importlib.util.spec_from_file_location(
    "dynamic_speculative_under_test",
    MODULE_PATH,
)
assert SPEC is not None and SPEC.loader is not None
DYNAMIC_SPECULATIVE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(DYNAMIC_SPECULATIVE)
build_dynamic_speculative_lookup = (
    DYNAMIC_SPECULATIVE.build_dynamic_speculative_lookup
)
validate_and_normalize_dynamic_speculative_schedule = (
    DYNAMIC_SPECULATIVE.validate_and_normalize_dynamic_speculative_schedule
)


class DynamicSpeculativeScheduleTests(unittest.TestCase):
    def test_builds_direct_batch_size_lookup(self):
        lookup = build_dynamic_speculative_lookup(
            [(1, 3, 3), (4, 8, 0)], 8, 3
        )
        self.assertEqual(lookup, [0, 3, 3, 3, 0, 0, 0, 0, 0])

    def test_gap_and_tail_carry_previous_k(self):
        lookup = build_dynamic_speculative_lookup(
            [(1, 2, 5), (5, 6, 2)], 8, 5
        )
        self.assertEqual(lookup, [0, 5, 5, 5, 5, 2, 2, 2, 2])

    def test_k_is_capped_by_configured_maximum(self):
        lookup = build_dynamic_speculative_lookup([(1, 8, 7)], 8, 3)
        self.assertEqual(lookup[1:], [3] * 8)

    def test_three_tier_schedule_and_zero_k_tail(self):
        lookup = build_dynamic_speculative_lookup(
            [(1, 3, 5), (4, 7, 3), (8, 8, 0)],
            max_batch_size=16,
            max_num_speculative_tokens=5,
        )
        self.assertEqual(lookup[1:4], [5, 5, 5])
        self.assertEqual(lookup[4:8], [3, 3, 3, 3])
        self.assertEqual(lookup[8:], [0] * 9)

    def test_rejects_overlap(self):
        with self.assertRaisesRegex(ValueError, "must not overlap"):
            validate_and_normalize_dynamic_speculative_schedule(
                [(1, 4, 3), (4, 8, 0)]
            )

    def test_rejects_non_one_start(self):
        with self.assertRaisesRegex(ValueError, "must start at batch size 1"):
            validate_and_normalize_dynamic_speculative_schedule([(2, 8, 3)])


if __name__ == "__main__":
    unittest.main()
