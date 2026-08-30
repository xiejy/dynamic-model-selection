"""Router behaviour, including the failure modes that make routing risky."""
import json

import pytest

from dms.client import Mode, ModelClient
from dms.pricing import PriceBook
from dms.replay import FixtureStore
from dms.routers.baseline import AlwaysRouter, RandomRouter
from dms.routers.heuristic import HeuristicRouter
from dms.routers.lexical import LexicalRouter, tokenise
from dms.routers.llm_classifier import LLMClassifierRouter, _parse_tier
from dms.strategies import CascadeStrategy, OracleStrategy
from dms.workload import Task, Workload

HAIKU = "claude-haiku-4-5"
OPUS = "claude-opus-5"


@pytest.fixture(scope="module")
def workload() -> Workload:
    return Workload.load()


@pytest.fixture
def client() -> ModelClient:
    return ModelClient(
        mode=Mode.SIMULATE, book=PriceBook.load(), store=FixtureStore("/tmp/unused")
    )


def _task(prompt: str, difficulty: str = "easy") -> Task:
    return Task(
        id="t", difficulty=difficulty, kind="k", prompt=prompt,
        expected="x", grader="exact_ci",
    )


# ------------------------------------------------------------------- baselines


def test_always_router_names_itself_after_its_model(client: ModelClient) -> None:
    router = AlwaysRouter(model=HAIKU)

    assert router.name == "always:haiku"
    assert router.route(_task("anything"), client).model == HAIKU


def test_always_router_costs_nothing_to_decide(client: ModelClient) -> None:
    assert AlwaysRouter(model=OPUS).route(_task("x"), client).spends == ()


def test_random_router_hits_its_target_fraction(
    client: ModelClient, workload: Workload
) -> None:
    router = RandomRouter(strong_model=OPUS, weak_model=HAIKU, strong_fraction=0.5)

    strong = sum(
        1 for task in workload if router.route(task, client).model == OPUS
    )

    assert 0.3 <= strong / len(workload) <= 0.7


def test_random_router_is_deterministic(client: ModelClient) -> None:
    router = RandomRouter(strong_model=OPUS, weak_model=HAIKU, strong_fraction=0.5)
    task = _task("stable")

    assert router.route(task, client).model == router.route(task, client).model


@pytest.mark.parametrize("fraction", [-0.1, 1.1])
def test_random_router_rejects_an_impossible_fraction(fraction: float) -> None:
    with pytest.raises(ValueError, match="strong_fraction"):
        RandomRouter(strong_model=OPUS, weak_model=HAIKU, strong_fraction=fraction)


def test_random_router_extremes_collapse_to_always(client: ModelClient) -> None:
    always_weak = RandomRouter(strong_model=OPUS, weak_model=HAIKU, strong_fraction=0.0)
    always_strong = RandomRouter(strong_model=OPUS, weak_model=HAIKU, strong_fraction=1.0)

    assert always_weak.route(_task("x"), client).model == HAIKU
    assert always_strong.route(_task("x"), client).model == OPUS


# ------------------------------------------------------------------- heuristic


def test_heuristic_costs_nothing_and_explains_itself(client: ModelClient) -> None:
    decision = HeuristicRouter().route(_task("Explain why this deadlock occurs"), client)

    assert decision.spends == ()
    assert "score=" in decision.why


def test_heuristic_separates_easy_from_hard(
    client: ModelClient, workload: Workload
) -> None:
    """It need not be perfect, but it must correlate with difficulty at all --
    otherwise it is a random router with extra steps."""
    router = HeuristicRouter()

    def strong_share(level: str) -> float:
        tasks = workload.by_difficulty(level)
        strong = sum(1 for t in tasks if router.route(t, client).model == OPUS)
        return strong / len(tasks)

    assert strong_share("hard") > strong_share("easy")


def test_heuristic_does_not_key_on_answer_format_instructions() -> None:
    """Regression guard. The first revision scored on 'answer with the number
    only', which appears at every difficulty, and so routed everything to Haiku."""
    router = HeuristicRouter()
    bare = "Trace this algorithm for four steps and report the final value."

    with_format, _ = router.score(bare + " Answer with the number only.")
    without_format, _ = router.score(bare)

    assert with_format == without_format


def test_heuristic_thresholds_are_configurable(client: ModelClient) -> None:
    everything_complex = HeuristicRouter(medium_threshold=-99, complex_threshold=-99)

    assert everything_complex.route(_task("hi"), client).model == OPUS


# --------------------------------------------------------------------- lexical


def test_tokenise_lowercases_and_drops_punctuation() -> None:
    assert tokenise("Explain the B-Tree, please!") == ["explain", "the", "b", "tree", "please"]


def test_lexical_router_fits_from_a_separate_exemplar_file() -> None:
    """Exemplars must not be the workload -- fitting on the eval set is how
    routing demos manufacture numbers that do not generalise."""
    router = LexicalRouter()

    assert len(router.exemplars) > 0
    assert {e.tier for e in router.exemplars} == {"simple", "medium", "complex"}


def test_lexical_router_matches_a_near_duplicate_exemplar(client: ModelClient) -> None:
    decision = LexicalRouter().route(
        _task("Extract the port number from this string. Answer with the number only."),
        client,
    )

    assert decision.model == HAIKU


def test_lexical_router_falls_back_to_the_capable_model_when_nothing_matches(
    client: ModelClient,
) -> None:
    """Out-of-distribution input must fail toward quality, not toward cost."""
    decision = LexicalRouter(min_similarity=0.99).route(_task("zzzz qqqq"), client)

    assert decision.model == OPUS
    assert "fallback" in decision.why


def test_lexical_router_reports_its_margin(client: ModelClient) -> None:
    decision = LexicalRouter().route(_task("Extract the port number."), client)

    assert "margin=" in decision.why


def test_lexical_router_rejects_an_empty_exemplar_file(tmp_path) -> None:
    empty = tmp_path / "empty.jsonl"
    empty.write_text("")

    with pytest.raises(ValueError, match="no exemplars"):
        LexicalRouter(exemplars_path=empty)


def test_lexical_router_costs_no_tokens(client: ModelClient) -> None:
    assert LexicalRouter().route(_task("Extract the port."), client).spends == ()


# -------------------------------------------------------------- llm classifier


@pytest.mark.parametrize(
    "text,expected",
    [("simple", "simple"), ("  Complex.", "complex"), ("MEDIUM", "medium"), ("banana", None)],
)
def test_tier_parsing(text: str, expected: str | None) -> None:
    assert _parse_tier(text) == expected


def test_llm_classifier_charges_its_own_tokens(client: ModelClient) -> None:
    """The only router here that spends to decide -- and it must show up."""
    decision = LLMClassifierRouter().route(_task("Explain the deadlock", "hard"), client)

    assert len(decision.spends) == 1
    assert decision.spends[0].role == "router"
    assert decision.spends[0].usage.total_tokens > 0


def test_llm_classifier_caps_its_own_output(client: ModelClient) -> None:
    """A router that writes an essay costs more than the model it is avoiding."""
    decision = LLMClassifierRouter().route(_task("Explain the deadlock", "hard"), client)

    assert decision.spends[0].usage.output_tokens <= 200


# ------------------------------------------------------------------ strategies


def test_cascade_without_verification_never_escalates(client: ModelClient) -> None:
    """Removing the verifier collapses the cascade into always-weak. The verifier
    is the whole mechanism, not an optimisation."""
    strategy = CascadeStrategy(verify=False)

    outcome = strategy.run(_task("anything", "hard"), client)

    assert outcome.escalated is False
    assert outcome.chosen_model == HAIKU


def test_cascade_pays_for_the_verification_on_every_task(client: ModelClient) -> None:
    outcome = CascadeStrategy().run(_task("anything", "hard"), client)

    assert any(spend.role == "verify" for spend in outcome.spends)


def test_oracle_charges_only_the_winning_call(client: ModelClient) -> None:
    outcome = OracleStrategy().run(_task("anything", "easy"), client)

    assert len(outcome.spends) == 1


def test_oracle_reports_failure_rather_than_faking_success(tmp_path) -> None:
    """When no model in the ladder can answer, the oracle must report the miss
    instead of claiming a win -- otherwise the ceiling is a fiction."""
    config = tmp_path / "sim.json"
    config.write_text(
        json.dumps(
            {
                "chars_per_token": 3.6,
                "accuracy_by_model_and_difficulty": {
                    model: {"easy": 0.0, "medium": 0.0, "hard": 0.0}
                    for model in ("claude-haiku-4-5", "claude-sonnet-5", "claude-opus-5")
                },
                "output_tokens_by_model": {},
                "latency_ms_by_model": {},
            }
        )
    )
    hopeless = ModelClient(
        mode=Mode.SIMULATE,
        book=PriceBook.load(),
        store=FixtureStore(tmp_path / "fx"),
        simulation_config=config,
    )

    outcome = OracleStrategy().run(_task("anything", "hard"), hopeless)

    assert outcome.correct is False
    assert "no model" in outcome.why
