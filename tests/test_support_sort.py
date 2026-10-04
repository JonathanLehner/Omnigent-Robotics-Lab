"""Regression checks for the rung-1 symbolic order planner."""

from unittest.mock import patch

from lab.pipeline import order_support_sort


def test_support_sort_ignores_spec_listing_order():
    spec = [
        {"id": "zeta", "type": "cube", "color": "red", "pos": [0.0, -0.1, 0.05], "yaw": 0.0, "on": []},
        {"id": "middle", "type": "cube", "color": "green", "pos": [0.0, -0.1, 0.15], "yaw": 0.0, "on": ["zeta"]},
        {"id": "alpha", "type": "cube", "color": "blue", "pos": [0.0, 0.1, 0.05], "yaw": 0.0, "on": []},
    ]

    # ponytail: spec listing order must not become an implicit topological-sort tie-break.
    with patch("lab.pipeline.sim.settle", return_value={"stable": True}):
        assert order_support_sort(spec, {}, {}) == order_support_sort(list(reversed(spec)), {}, {})


if __name__ == "__main__":
    test_support_sort_ignores_spec_listing_order()
