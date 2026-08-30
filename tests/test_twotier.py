"""Binary dispatch evaluation: routability, significance, and the verdict logic."""
import pytest

from dms.twotier import mcnemar_p, routability, tasks_needed, two_tier_map


def test_two_tier_map_sends_anything_above_simple_to_the_high_model() -> None:
    """Ambiguity must resolve toward quality, not toward cost."""
    tiers = two_tier_map(low="LOW", high="HIGH")

    assert tiers["simple"] == "LOW"
    assert tiers["medium"] == "HIGH"
    assert tiers["complex"] == "HIGH"


# ------------------------------------------------------------------ routability


def test_full_agreement_means_nothing_is_routable() -> None:
    """If the models never disagree, a dispatcher cannot change quality at all --
    only cost. This is the ceiling that no router improvement can raise."""
    same = {"a": True, "b": True, "c": False}

    result = routability(same, same)

    assert result.routable == 0
    assert result.routable_share == 0.0
    assert result.agreement == 1.0


def test_routable_share_counts_disagreements_in_both_directions() -> None:
    low = {"a": True, "b": False, "c": True, "d": False}
    high = {"a": True, "b": True, "c": False, "d": False}

    result = routability(low, high)

    assert result.high_only == ("b",)   # dispatcher must catch this one
    assert result.low_only == ("c",)    # the expensive model was simply wrong
    assert result.routable_share == 0.5
    assert result.both_right == 1 and result.both_wrong == 1


def test_low_only_tasks_are_reported_separately() -> None:
    """Tasks the EXPENSIVE model gets wrong are not a router failure, and
    lumping them in would blame the dispatcher for the model's own miss."""
    low = {"a": True}
    high = {"a": False}

    assert routability(low, high).low_only == ("a",)


# ------------------------------------------------------------------ significance


def test_identical_strategies_are_never_significant() -> None:
    outcomes = {"a": True, "b": False, "c": True}

    _, _, p = mcnemar_p(outcomes, outcomes)

    assert p == 1.0


def test_a_handful_of_discordant_tasks_cannot_reach_significance() -> None:
    """The core statistical point: 5 wins and 2 losses looks decisive and is not.
    This is exactly the shape of the measured Opus-vs-Haiku comparison."""
    a = {f"t{i}": True for i in range(36)}
    b = dict(a)
    for i in range(5):
        b[f"t{i}"] = False   # a wins 5
    for i in range(5, 7):
        a[f"t{i}"] = False   # b wins 2

    wins, losses, p = mcnemar_p(a, b)

    assert (wins, losses) == (5, 2)
    assert p > 0.05


def test_a_clean_sweep_does_reach_significance() -> None:
    a = {f"t{i}": True for i in range(36)}
    b = dict(a)
    for i in range(8):
        b[f"t{i}"] = False

    _, _, p = mcnemar_p(a, b)

    assert p < 0.05


def test_sample_size_grows_as_the_effect_shrinks() -> None:
    assert tasks_needed(5.0) < tasks_needed(2.0)


def test_this_suite_is_far_too_small_to_resolve_a_five_point_effect() -> None:
    """36 tasks is enough to show mechanism. It is not enough to rank routers."""
    assert tasks_needed(5.0, discordance=0.19) > 300


@pytest.mark.parametrize("discordance", [0.05, 0.19, 0.40])
def test_sample_size_is_finite_across_plausible_disagreement_rates(discordance) -> None:
    assert 0 < tasks_needed(5.0, discordance) < 100_000
