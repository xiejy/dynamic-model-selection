"""Workload integrity and grader behaviour."""
import pytest

from dms.grading import GRADERS, grade, normalise
from dms.workload import DIFFICULTIES, Workload


@pytest.fixture(scope="module")
def workload() -> Workload:
    return Workload.load()


# --------------------------------------------------------------------------- workload


def test_workload_loads_and_is_not_trivially_small(workload: Workload) -> None:
    assert len(workload) >= 30


def test_every_difficulty_tier_is_populated(workload: Workload) -> None:
    """Routing has nothing to exploit unless difficulty genuinely varies."""
    mix = workload.mix()

    assert set(mix) == set(DIFFICULTIES)
    assert all(count >= 10 for count in mix.values()), mix


def test_task_ids_are_unique(workload: Workload) -> None:
    ids = [task.id for task in workload]

    assert len(ids) == len(set(ids))


def test_every_task_names_a_grader_that_exists(workload: Workload) -> None:
    unknown = {task.grader for task in workload} - GRADERS.keys()

    assert not unknown, f"workload references undefined graders: {unknown}"


def test_every_task_is_graded_correct_by_its_own_expected_answer(workload: Workload) -> None:
    """A task whose own gold answer fails its grader is a broken task -- it would
    silently cap every model's score and make the whole benchmark meaningless."""
    broken = [
        task.id
        for task in workload
        if not grade(task.expected, task.expected, task.grader)
    ]

    assert not broken, f"tasks that fail their own gold answer: {broken}"


def test_workload_rejects_an_unknown_difficulty(tmp_path) -> None:
    path = tmp_path / "bad.jsonl"
    path.write_text(
        '{"id":"x","difficulty":"trivial","kind":"k","prompt":"p",'
        '"expected":"e","grader":"exact_ci"}\n'
    )

    with pytest.raises(ValueError, match="difficulty"):
        Workload.load(path)


def test_workload_rejects_duplicate_ids(tmp_path) -> None:
    line = (
        '{"id":"x","difficulty":"easy","kind":"k","prompt":"p",'
        '"expected":"e","grader":"exact_ci"}'
    )
    path = tmp_path / "dupe.jsonl"
    path.write_text(f"{line}\n{line}\n")

    with pytest.raises(ValueError, match="duplicate"):
        Workload.load(path)


# --------------------------------------------------------------------------- graders


@pytest.mark.parametrize(
    "answer,expected,ok",
    [
        ("ERROR", "ERROR", True),
        ("  error. ", "ERROR", True),
        ("`503`", "503", True),
        ("WARN", "ERROR", False),
    ],
)
def test_exact_ci_is_forgiving_about_formatting_only(answer, expected, ok) -> None:
    assert grade(answer, expected, "exact_ci") is ok


@pytest.mark.parametrize(
    "answer,ok",
    [
        ("yes", True),
        ("Yes, it is idempotent.", True),
        ("no", False),
        ("It depends on the implementation", False),  # ambiguous is not a pass
    ],
)
def test_yesno_requires_an_actual_polarity(answer, ok) -> None:
    assert grade(answer, "yes", "yesno") is ok


def test_contains_any_accepts_alternative_phrasings() -> None:
    expected = "mutable default|shared default|default argument"

    assert grade("It uses a mutable default argument", expected, "contains_any") is True
    assert grade("the list is global", expected, "contains_any") is False


def test_sql_shape_requires_every_fragment() -> None:
    expected = "select|count|from orders|group by|customer_id"
    good = "SELECT customer_id, COUNT(*) FROM orders GROUP BY customer_id;"

    assert grade(good, expected, "sql_shape") is True
    assert grade("SELECT * FROM orders;", expected, "sql_shape") is False


def test_cron_ignores_surrounding_prose_and_spacing() -> None:
    assert grade("`0 3 * * 1`", "0 3 * * 1", "cron") is True
    assert grade("The expression is 0 3 * * 1 weekly", "0 3 * * 1", "cron") is True
    assert grade("0 4 * * 1", "0 3 * * 1", "cron") is False


def test_grader_strips_markdown_code_fences() -> None:
    assert normalise("```sql\nSELECT 1\n```") == "select 1"


def test_unknown_grader_is_a_loud_configuration_error() -> None:
    with pytest.raises(KeyError, match="unknown grader"):
        grade("a", "a", "vibes")
