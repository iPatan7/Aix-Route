"""The `aix_route` public import surface must stay stable for `pip install aix-route`."""

from __future__ import annotations


def test_top_level_imports() -> None:
    from aix_route import (  # noqa: F401
        delegation_decision,
        evaluate,
        fit_horizon,
        horizon_table,
        recommend_model,
        should_delegate,
        should_delegate_batch,
    )


def test_submodule_paths() -> None:
    from aix_route.calibrate import empirical_d_star, fit_horizon  # noqa: F401
    from aix_route.policy import horizon_table  # noqa: F401
    from aix_route.tasks import PermutationTask  # noqa: F401


def test_version_matches_implementation() -> None:
    import aix_route
    import deterministic_horizon

    assert aix_route.__version__ == deterministic_horizon.__version__


def test_aix_route_and_dh_share_policy() -> None:
    # Both names resolve to the same policy module — one source of truth.
    import aix_route.policy as ar
    import deterministic_horizon.policy as dh

    assert ar is dh
    assert aix_route_decision() == dh.should_delegate(35, "gpt-4o")


def aix_route_decision() -> bool:
    from aix_route import should_delegate

    return should_delegate(35, "gpt-4o")
