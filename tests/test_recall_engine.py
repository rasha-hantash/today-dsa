"""Tests for the recall engine.

These tests double as the spec: each test name is a complete sentence describing
a user-visible behavior of the snapshot-mode SM-2 lite recall queue.
"""

from __future__ import annotations

import json
from datetime import date, timedelta
from pathlib import Path

import pytest

from recall_engine import (
    BehavioralTopic,
    CategoryProgress,
    HardestMark,
    Mock,
    MockPrereq,
    Phase,
    Problem,
    Readiness,
    RecallItem,
    SDChapter,
    Touch,
    aggregate_touches,
    append_to_hardest_ledger,
    append_to_ledger,
    avg_new_per_day,
    compute_new,
    compute_readiness,
    compute_recall,
    current_phase,
    day_n_for,
    due_date,
    interval_for,
    apply_mock_updates,
    complete_mock,
    compute_maintenance,
    ensure_runtime_dirs,
    init_curriculum_file,
    min_touches_in_scope,
    schedule_mock,
    load_hardest_ledger,
    load_ledger,
    mock_prereq_status,
    next_sd_chapter,
    overdue_days,
    parse_behavioral,
    parse_completions,
    parse_curriculum,
    parse_curriculum_dsa_state,
    parse_hardest_marks,
    parse_mock_updates,
    parse_mocks,
    parse_phases,
    parse_sd_chapters,
    projected_end_date,
    recompute,
    render_readiness_block,
    render_today,
    start_date,
    write_curriculum_dsa,
    write_curriculum_mocks,
)


# ─── SM-2 lite interval expansion ──────────────────────────────────────────────


@pytest.mark.parametrize(
    "touches,expected_interval_days",
    [(1, 1), (2, 3), (3, 7), (4, 21), (5, 21), (6, 21), (10, 21)],
    ids=[
        "a problem solved once is due 1 day later",
        "a problem solved twice is due 3 days later",
        "a problem solved three times is due 7 days later",
        "a problem solved four times caps at 21-day mastery interval",
        "a fifth solve does not extend the interval past the 21-day cap",
        "the sixth solve does not extend the interval past the 21-day cap",
        "extra solves past four never push the interval past the 21-day cap",
    ],
)
def test_sm2_lite_interval_expansion(touches: int, expected_interval_days: int) -> None:
    assert interval_for(touches) == expected_interval_days


# ─── Overdue arithmetic ────────────────────────────────────────────────────────


@pytest.mark.parametrize(
    "touches,last_touched,today,expected_overdue",
    [
        (1, date(2026, 5, 6), date(2026, 5, 6), -1),
        (1, date(2026, 5, 6), date(2026, 5, 7), 0),
        (1, date(2026, 5, 6), date(2026, 5, 8), 1),
        (4, date(2026, 5, 1), date(2026, 5, 22), 0),
        (4, date(2026, 5, 1), date(2026, 6, 1), 10),
    ],
    ids=[
        "a problem solved today is not yet due tomorrow",
        "a problem due today reads as zero days overdue",
        "a once-solved problem skipped one day reads as one day overdue",
        "a four-times-mastered problem becomes due exactly twenty-one days later",
        "a four-times-mastered problem can be ignored three weeks before going overdue",
    ],
)
def test_overdue_days_calculation(
    touches: int, last_touched: date, today: date, expected_overdue: int
) -> None:
    assert overdue_days(touches, last_touched, today) == expected_overdue


def test_due_date_is_last_touched_plus_sm2_interval() -> None:
    assert due_date(1, date(2026, 5, 6)) == date(2026, 5, 7)
    assert due_date(3, date(2026, 5, 6)) == date(2026, 5, 13)


# ─── Touch aggregation ─────────────────────────────────────────────────────────


def test_aggregate_touches_counts_completions_per_problem() -> None:
    ledger = [
        Touch("[A] Two Sum", date(2026, 5, 1)),
        Touch("[A] Two Sum", date(2026, 5, 4)),
        Touch("[A] Group Anagrams", date(2026, 5, 6)),
    ]
    assert aggregate_touches(ledger) == {
        "[A] Two Sum": (2, date(2026, 5, 4)),
        "[A] Group Anagrams": (1, date(2026, 5, 6)),
    }


def test_aggregate_touches_tracks_only_the_latest_completion_date() -> None:
    """If the user solves the same problem twice on different days, the schedule
    should reset from the most recent completion, not the first."""
    ledger = [
        Touch("[A] Two Sum", date(2026, 5, 1)),
        Touch("[A] Two Sum", date(2026, 5, 8)),
        Touch("[A] Two Sum", date(2026, 5, 4)),  # out-of-order entry
    ]
    aggregated = aggregate_touches(ledger)
    assert aggregated["[A] Two Sum"] == (3, date(2026, 5, 8))


# ─── Recall ranking and filtering ──────────────────────────────────────────────


def test_recall_ranks_the_most_overdue_problem_first() -> None:
    ledger = [
        Touch("[A] Two Sum", date(2026, 5, 1)),  # 1× → due May 2 → 5d overdue on May 7
        Touch("[A] Group Anagrams", date(2026, 5, 5)),  # 1× → due May 6 → 1d overdue
        Touch("[A] Valid Anagram", date(2026, 5, 6)),  # 1× → due May 7 → 0d overdue
    ]
    recall = compute_recall(ledger, today=date(2026, 5, 7), limit=10)
    assert [r.problem for r in recall] == [
        "[A] Two Sum",
        "[A] Group Anagrams",
        "[A] Valid Anagram",
    ]


def test_recall_caps_at_the_requested_limit_when_more_items_are_overdue() -> None:
    ledger = [Touch(f"[X] P{i}", date(2026, 5, 1)) for i in range(15)]
    recall = compute_recall(ledger, today=date(2026, 5, 7), limit=10)
    assert len(recall) == 10


def test_recall_excludes_problems_whose_due_date_has_not_yet_arrived() -> None:
    ledger = [
        Touch("[A] Two Sum", date(2026, 5, 4)),
        Touch("[A] Two Sum", date(2026, 5, 6)),  # 2× → 3d interval → due May 9
    ]
    recall = compute_recall(ledger, today=date(2026, 5, 7), limit=10)
    assert recall == []


def test_recall_is_empty_when_no_problem_has_ever_been_touched() -> None:
    assert compute_recall(ledger=[], today=date(2026, 5, 7), limit=10) == []


def test_recall_item_carries_metadata_for_rendering() -> None:
    ledger = [Touch("[A] Two Sum", date(2026, 5, 1))]
    recall = compute_recall(ledger, today=date(2026, 5, 7), limit=10)
    [item] = recall
    assert item.problem == "[A] Two Sum"
    assert item.touches == 1
    assert item.last_touched == date(2026, 5, 1)
    assert item.days_overdue == 5


# ─── Min-touches trigger + Maintenance (round-robin) selection ────────────────


def _easy(text: str) -> Problem:
    return Problem(text, difficulty="E")


def _medium(text: str) -> Problem:
    return Problem(text, difficulty="M")


def _hard(text: str) -> Problem:
    return Problem(text, difficulty="H")


def _five_touches(problem: str, last: date = date(2026, 5, 1)) -> list[Touch]:
    return [Touch(problem, last - timedelta(days=n)) for n in range(5)]


def test_min_touches_returns_zero_when_any_in_scope_problem_is_untouched() -> None:
    curriculum = [_easy("[A] Two Sum"), _easy("[A] 3Sum")]
    ledger = _five_touches("[A] Two Sum")
    assert min_touches_in_scope(curriculum, ledger) == 0


def test_min_touches_returns_minimum_touch_count_across_in_scope_problems() -> None:
    curriculum = [_easy("[A] Two Sum"), _easy("[A] 3Sum")]
    ledger = _five_touches("[A] Two Sum") + [
        Touch("[A] 3Sum", date(2026, 5, 1)),
        Touch("[A] 3Sum", date(2026, 5, 2)),
        Touch("[A] 3Sum", date(2026, 5, 3)),
    ]
    assert min_touches_in_scope(curriculum, ledger) == 3


def test_min_touches_ignores_problems_outside_the_requested_difficulties() -> None:
    """E/M scope sees only the well-touched Easies; an untouched Hard does
    not pull the minimum down to zero."""
    curriculum = [_easy("[A] Two Sum"), _hard("[A] N-Queens")]
    ledger = _five_touches("[A] Two Sum")
    assert min_touches_in_scope(curriculum, ledger, difficulties=("E", "M")) == 5


def test_compute_maintenance_returns_empty_until_every_in_scope_problem_is_mastered() -> None:
    curriculum = [_easy("[A] Two Sum"), _easy("[A] 3Sum")]
    ledger = (
        _five_touches("[A] Two Sum")
        + [Touch("[A] 3Sum", date(2026, 5, d)) for d in (1, 3, 5)]  # only 3 touches
    )
    items = compute_maintenance(curriculum, ledger, today=date(2026, 5, 30))
    assert items == []


def test_compute_maintenance_interleaves_patterns_in_round_robin_order() -> None:
    """When every in-scope problem is mastered, the section deals one problem
    from each pattern in rotation rather than draining one pattern at a time."""
    curriculum = [
        _easy("[Arrays] A1"), _easy("[Arrays] A2"), _easy("[Arrays] A3"),
        _easy("[Trees] T1"), _easy("[Trees] T2"),
        _easy("[Graphs] G1"),
    ]
    ledger: list[Touch] = []
    for p in curriculum:
        ledger += _five_touches(p.text, last=date(2026, 5, 30))
    items = compute_maintenance(
        curriculum, ledger, today=date(2026, 6, 30), limit=6
    )
    # Largest bucket (Arrays, 3) first per round, then Trees (2), then Graphs (1);
    # second round drains the remaining Arrays/Trees items.
    patterns = [it.problem.split("]")[0] + "]" for it in items]
    assert patterns == ["[Arrays]", "[Trees]", "[Graphs]", "[Arrays]", "[Trees]", "[Arrays]"]


def test_compute_maintenance_orders_least_recently_touched_first_within_a_pattern() -> None:
    curriculum = [_easy("[A] Two Sum"), _easy("[A] 3Sum")]
    # Both mastered, but Two Sum was touched more recently than 3Sum.
    ledger = (
        _five_touches("[A] Two Sum", last=date(2026, 5, 30))
        + _five_touches("[A] 3Sum", last=date(2026, 5, 20))
    )
    items = compute_maintenance(curriculum, ledger, today=date(2026, 6, 30), limit=2)
    assert [it.problem for it in items] == ["[A] 3Sum", "[A] Two Sum"]


def test_compute_maintenance_caps_at_the_requested_limit() -> None:
    curriculum = [_easy(f"[A] P{i}") for i in range(20)]
    ledger: list[Touch] = []
    for p in curriculum:
        ledger += _five_touches(p.text)
    items = compute_maintenance(curriculum, ledger, today=date(2026, 6, 30), limit=7)
    assert len(items) == 7


def test_render_today_omits_maintenance_section_when_no_items_to_surface() -> None:
    out = render_today(
        today=date(2026, 5, 11),
        recall=[],
        new=[],
        maintenance=None,
    )
    assert "## Maintenance" not in out


def test_render_today_places_maintenance_section_below_todays_hardest() -> None:
    """Maintenance is ambient/optional; the day's prescribed work (Recall,
    New, SD, Hardest pick) sits above it. A reader scrolling top-down hits
    obligation before rotation."""
    items = [
        RecallItem(
            problem="[Arrays] Two Sum",
            touches=4,
            last_touched=date(2026, 5, 1),
            days_overdue=-10,
            difficulty="E",
        ),
    ]
    out = render_today(
        today=date(2026, 5, 11),  # a Monday — Mon-Fri renders the hardest pick block
        recall=[],
        new=[Problem("[Arrays] -> 3Sum", difficulty="M", phase=1)],
        maintenance=items,
    )
    hardest_idx = out.find("## Today's hardest")
    maintenance_idx = out.find("## Maintenance")
    assert hardest_idx != -1, "expected Mon-Fri hardest pick section"
    assert maintenance_idx != -1, "expected Maintenance section"
    assert maintenance_idx > hardest_idx


def test_render_today_renders_maintenance_section_when_items_present() -> None:
    items = [
        RecallItem(
            problem="[Arrays] Two Sum",
            touches=4,
            last_touched=date(2026, 5, 1),
            days_overdue=-10,
            difficulty="E",
        ),
        RecallItem(
            problem="[Trees] Invert Binary Tree",
            touches=4,
            last_touched=date(2026, 5, 2),
            days_overdue=-9,
            difficulty="E",
        ),
    ]
    out = render_today(
        today=date(2026, 5, 11),
        recall=[],
        new=[],
        maintenance=items,
    )
    assert "## Maintenance — interleaved across patterns" in out
    assert "[Arrays] Two Sum" in out
    assert "[Trees] Invert Binary Tree" in out
    # Maintenance lines do NOT show "Xd overdue" — that's a Recall-only annotation.
    maintenance_block = out.split("## Maintenance")[1]
    assert "overdue" not in maintenance_block


def test_compute_maintenance_excludes_problems_outside_the_difficulty_scope() -> None:
    """Hards stay out of the Maintenance bucket even when they're well-touched."""
    curriculum = [
        _easy("[A] Two Sum"),
        _easy("[A] 3Sum"),
        _hard("[A] N-Queens"),
    ]
    ledger: list[Touch] = []
    for p in curriculum:
        ledger += _five_touches(p.text)
    items = compute_maintenance(curriculum, ledger, today=date(2026, 6, 30), limit=10)
    assert all(it.difficulty in ("E", "M") for it in items)
    assert "[A] N-Queens" not in [it.problem for it in items]


# ─── New (next-up) selection ───────────────────────────────────────────────────


CURRICULUM = [
    Problem("[A] P1"),
    Problem("[A] P2"),
    Problem("[A] P3"),
    Problem("[A] P4"),
    Problem("[A] P5"),
]


def test_new_picks_the_first_three_unchecked_source_day_problems_in_order() -> None:
    new = compute_new(CURRICULUM, ledger=[], limit=3)
    assert [p.text for p in new] == ["[A] P1", "[A] P2", "[A] P3"]


def test_new_advances_past_problems_already_in_the_ledger() -> None:
    ledger = [
        Touch("[A] P1", date(2026, 5, 6)),
        Touch("[A] P3", date(2026, 5, 6)),
    ]
    new = compute_new(CURRICULUM, ledger, limit=3)
    assert [p.text for p in new] == ["[A] P2", "[A] P4", "[A] P5"]


def test_new_surfaces_yesterdays_skipped_problems_first_in_document_order() -> None:
    """If you skip Day 1 problems, the next morning they appear in New ahead of
    Day 2 — no manual rescheduling needed."""
    ledger = [Touch("[A] P1", date(2026, 5, 6))]
    new = compute_new(CURRICULUM, ledger, limit=3)
    assert [p.text for p in new] == ["[A] P2", "[A] P3", "[A] P4"]


def test_new_returns_fewer_than_limit_when_curriculum_is_exhausted() -> None:
    ledger = [Touch(p.text, date(2026, 5, 6)) for p in CURRICULUM[:-1]]
    new = compute_new(CURRICULUM, ledger, limit=3)
    assert [p.text for p in new] == ["[A] P5"]


# ─── Curriculum parser ─────────────────────────────────────────────────────────


SAMPLE_CURRICULUM_MD = """\
# NeetCode Curriculum

Source of truth — by-phase, by-pattern list.

## Legend

- `(E)` `(M)` `(H)` — LeetCode difficulty.

---

## DSA

### Phase 1 — Linear Patterns E+M (5 new/day)

#### Arrays & Hashing

- [ ] Contains Duplicate (E)
- [ ] Valid Anagram (E)
- [ ] Two Sum (E)

### Phase 5 — Hard Problems (2 new/day)

#### Two Pointers

- [ ] Trapping Rain Water `T2` (H)
"""


def test_curriculum_parser_extracts_every_problem_under_pattern_headings() -> None:
    problems = parse_curriculum(SAMPLE_CURRICULUM_MD)
    assert [p.text for p in problems] == [
        "[Arrays & Hashing] -> Contains Duplicate",
        "[Arrays & Hashing] -> Valid Anagram",
        "[Arrays & Hashing] -> Two Sum",
        "[Two Pointers] -> Trapping Rain Water",
    ]


def test_curriculum_parser_strips_the_T2_marker_from_canonical_text() -> None:
    """The vestigial `T2` marker on a few problems must not contaminate the
    canonical key — otherwise the same problem would appear twice in the ledger."""
    problems = parse_curriculum(SAMPLE_CURRICULUM_MD)
    trapping = next(p for p in problems if "Trapping" in p.text)
    assert trapping.text == "[Two Pointers] -> Trapping Rain Water"


def test_curriculum_parser_ignores_legend_section() -> None:
    """The Legend section has bullet lines that shouldn't parse as problems."""
    problems = parse_curriculum(SAMPLE_CURRICULUM_MD)
    texts = {p.text for p in problems}
    assert not any("LeetCode difficulty" in t for t in texts)


# ─── Completion parser ────────────────────────────────────────────────────────


def test_completion_parser_captures_problems_with_a_done_date_stamp() -> None:
    md = (
        "- [x] [A] -> Two Sum ✅ 2026-05-06\n"
        "- [x] [A] -> Group Anagrams ✅ 2026-05-06\n"
        "- [ ] [A] -> Valid Anagram\n"
    )
    assert parse_completions(md) == [
        Touch("[A] -> Two Sum", date(2026, 5, 6)),
        Touch("[A] -> Group Anagrams", date(2026, 5, 6)),
    ]


def test_completion_parser_ignores_unchecked_lines() -> None:
    assert parse_completions("- [ ] [A] -> P1") == []


def test_completion_parser_ignores_checked_lines_without_a_date_stamp() -> None:
    """Without a date we cannot schedule the next review — better to skip silently
    than to invent a date."""
    assert parse_completions("- [x] [A] -> P1") == []


def test_completion_parser_strips_metadata_suffix_from_canonical_text() -> None:
    """today.md adds an em-dash annotation like ` — 1d overdue · 1× · last May 6`.
    The canonical key must match the curriculum form, not the annotated render."""
    md = "- [x] [A] -> Two Sum — 1d overdue · 1× · last May 6 ✅ 2026-05-07"
    assert parse_completions(md) == [Touch("[A] -> Two Sum", date(2026, 5, 7))]


def test_completion_parser_strips_source_day_annotation() -> None:
    """The New section appends ` (Day N)` for context — strip it for the ledger."""
    md = "- [x] [A] -> Valid Anagram (Day 1) ✅ 2026-05-06"
    assert parse_completions(md) == [Touch("[A] -> Valid Anagram", date(2026, 5, 6))]


def test_completion_parser_strips_T2_marker() -> None:
    md = "- [x] [Two Pointers] -> Trapping Rain Water `T2` ✅ 2026-05-09"
    assert parse_completions(md) == [
        Touch("[Two Pointers] -> Trapping Rain Water", date(2026, 5, 9))
    ]


# ─── Ledger I/O ──────────────────────────────────────────────────────────────


def test_ledger_round_trips_a_single_touch(tmp_path: Path) -> None:
    path = tmp_path / "completions.jsonl"
    append_to_ledger(path, [Touch("[A] Two Sum", date(2026, 5, 6))])
    assert load_ledger(path) == [Touch("[A] Two Sum", date(2026, 5, 6))]


def test_ledger_records_separate_touches_on_different_days(tmp_path: Path) -> None:
    """Two valid completions of the same problem on different days must both land
    in the ledger so the touch counter can grow."""
    path = tmp_path / "completions.jsonl"
    append_to_ledger(
        path,
        [
            Touch("[A] Two Sum", date(2026, 5, 6)),
            Touch("[A] Two Sum", date(2026, 5, 9)),
        ],
    )
    assert len(load_ledger(path)) == 2


def test_ledger_dedupes_a_repeated_completion_on_the_same_day(tmp_path: Path) -> None:
    """Re-running recompute on a today.md that hasn't changed should not double-log
    yesterday's completions."""
    path = tmp_path / "completions.jsonl"
    append_to_ledger(path, [Touch("[A] Two Sum", date(2026, 5, 6))])
    appended = append_to_ledger(path, [Touch("[A] Two Sum", date(2026, 5, 6))])
    assert appended == 0
    assert load_ledger(path) == [Touch("[A] Two Sum", date(2026, 5, 6))]


def test_ledger_returns_empty_list_when_no_file_exists(tmp_path: Path) -> None:
    """First-run case — no completions yet, no file yet."""
    assert load_ledger(tmp_path / "does-not-exist.jsonl") == []


# ─── End-to-end recompute ────────────────────────────────────────────────────


THREE_DAY_CURRICULUM_MD = """\
## DSA

### Phase 1 — Linear (5 new/day)

#### A

- [ ] P1 (E)
- [ ] P2 (E)
- [ ] P3 (E)
- [ ] P4 (E)
- [ ] P5 (E)
"""


def test_recompute_creates_today_md_on_first_run(tmp_path: Path) -> None:
    """Day 1 morning: nothing exists yet. Recompute generates today's set."""
    daily = tmp_path / "curriculum.md"
    daily.write_text(THREE_DAY_CURRICULUM_MD)
    today_md = tmp_path / "today.md"
    ledger = tmp_path / "completions.jsonl"

    result = recompute(daily, today_md, ledger, today=date(2026, 5, 6))

    assert today_md.exists()
    text = today_md.read_text()
    assert "[[a|A]] -> P1" in text
    assert "[[a|A]] -> P2" in text
    assert result.new_size >= 1
    assert result.recall_size == 0
    assert result.new_touches_logged == 0


def test_recompute_logs_yesterday_completions_into_the_ledger(tmp_path: Path) -> None:
    """Tomorrow morning: previous today.md has a checked, dated item. Recompute
    folds that touch into the ledger before regenerating today.md."""
    daily = tmp_path / "curriculum.md"
    daily.write_text(THREE_DAY_CURRICULUM_MD)
    ledger = tmp_path / "completions.jsonl"
    today_md = tmp_path / "today.md"
    today_md.write_text("- [x] [A] -> P1 (Day 1) ✅ 2026-05-06\n")

    result = recompute(daily, today_md, ledger, today=date(2026, 5, 7))

    assert result.new_touches_logged == 1
    assert load_ledger(ledger) == [Touch("[A] -> P1", date(2026, 5, 6))]


def test_recompute_surfaces_yesterdays_completion_into_recall_the_next_day(
    tmp_path: Path,
) -> None:
    """A problem solved on May 6 with 1 touch is due May 7 — it should appear in
    the May 7 Recall section after recompute folds the touch into the ledger."""
    daily = tmp_path / "curriculum.md"
    daily.write_text(THREE_DAY_CURRICULUM_MD)
    ledger = tmp_path / "completions.jsonl"
    today_md = tmp_path / "today.md"
    today_md.write_text("- [x] [A] -> P1 (Day 1) ✅ 2026-05-06\n")

    recompute(daily, today_md, ledger, today=date(2026, 5, 7))

    text = today_md.read_text()
    assert "## Recall" in text
    # P1 is now overdue 0 days (due exactly today) and should appear in Recall.
    # The recall line wikilinks the pattern and (when curriculum has a link)
    # hyperlinks the name — the test curriculum supplies no link so the name
    # stays plain.
    recall_section = text.split("## New")[0]
    assert "[[a|A]] -> P1" in recall_section


def test_recompute_does_not_relog_when_run_twice_with_no_new_completions(
    tmp_path: Path,
) -> None:
    """Snapshot semantics: rerunning recompute mid-day must not corrupt the
    ledger. Each completion gets logged exactly once regardless of how many
    times the user triggers a refresh."""
    daily = tmp_path / "curriculum.md"
    daily.write_text(THREE_DAY_CURRICULUM_MD)
    ledger = tmp_path / "completions.jsonl"
    today_md = tmp_path / "today.md"
    today_md.write_text("- [x] [A] -> P1 (Day 1) ✅ 2026-05-06\n")

    first = recompute(daily, today_md, ledger, today=date(2026, 5, 7))
    # User did not check anything else; a manual mid-day re-run should be a no-op
    # for the ledger. (today.md is rewritten but contains the same canonical set.)
    second = recompute(daily, today_md, ledger, today=date(2026, 5, 7))

    assert first.new_touches_logged == 1
    assert second.new_touches_logged == 0
    assert len(load_ledger(ledger)) == 1


def test_recompute_advances_new_section_past_completed_curriculum_problems(
    tmp_path: Path,
) -> None:
    """After P1 is logged as solved, today's New section should not surface P1
    again — it advances to the next unsolved curriculum problem."""
    daily = tmp_path / "curriculum.md"
    daily.write_text(THREE_DAY_CURRICULUM_MD)
    ledger = tmp_path / "completions.jsonl"
    today_md = tmp_path / "today.md"
    today_md.write_text("- [x] [A] -> P1 (Day 1) ✅ 2026-05-06\n")

    recompute(daily, today_md, ledger, today=date(2026, 5, 7))

    text = today_md.read_text()
    new_section = text.split("## New")[1] if "## New" in text else ""
    assert "[A] -> P1" not in new_section


# ─── Renderer ────────────────────────────────────────────────────────────────


def test_renderer_produces_an_empty_recall_message_when_nothing_is_overdue() -> None:
    out = render_today(today=date(2026, 5, 6), recall=[], new=[Problem("[A] P1", 1)])
    assert "## Recall" in out
    assert "## New" in out
    # Some explicit empty-state copy so the user knows the section is intentionally empty.
    assert "Empty" in out or "empty" in out


def test_renderer_includes_a_dated_header_for_orientation() -> None:
    out = render_today(today=date(2026, 5, 6), recall=[], new=[])
    assert "May 6" in out or "2026-05-06" in out


# ─── Difficulty + source tags ─────────────────────────────────────────────────


_TAGGED_DAILY_MD = """\
## DSA

### Phase 1 — Linear (5 new/day)

#### Arrays & Hashing

- [ ] Contains Duplicate (E)
- [ ] Group Anagrams (M)
- [ ] Customers With 2-Day 2-Page Visits (M) (company question)

### Phase 5 — Hards (2 new/day)

#### Two Pointers

- [ ] Trapping Rain Water (H)

#### Segment Tree

- [ ] Count of Smaller After Self (H) (nc-150+)

### Phase 7 — NC-150+ (3 new/day)

#### Boyer-Moore

- [ ] Majority Element II (M) (nc-150+)
"""


def test_curriculum_parser_extracts_difficulty_marker() -> None:
    problems = parse_curriculum(_TAGGED_DAILY_MD)
    by_text = {p.text: p for p in problems}
    assert by_text["[Arrays & Hashing] -> Contains Duplicate"].difficulty == "E"
    assert by_text["[Arrays & Hashing] -> Group Anagrams"].difficulty == "M"
    assert by_text["[Two Pointers] -> Trapping Rain Water"].difficulty == "H"


def test_curriculum_parser_defaults_source_to_nc_150_when_no_source_tag() -> None:
    """NC150 problems carry only a difficulty marker; their source is implicitly 'nc-150'."""
    problems = parse_curriculum(_TAGGED_DAILY_MD)
    by_text = {p.text: p for p in problems}
    assert by_text["[Arrays & Hashing] -> Contains Duplicate"].source == "nc-150"


def test_curriculum_parser_extracts_explicit_source_marker() -> None:
    problems = parse_curriculum(_TAGGED_DAILY_MD)
    by_text = {p.text: p for p in problems}
    assert by_text["[Boyer-Moore] -> Majority Element II"].source == "nc-150+"
    assert by_text["[Segment Tree] -> Count of Smaller After Self"].source == "nc-150+"
    assert (
        by_text["[Arrays & Hashing] -> Customers With 2-Day 2-Page Visits"].source
        == "company question"
    )


def test_curriculum_parser_canonical_text_omits_difficulty_and_source_tags() -> None:
    """Ledger keys must match across renders, so the canonical text excludes
    annotations the renderer might add or drop."""
    problems = parse_curriculum(_TAGGED_DAILY_MD)
    texts = {p.text for p in problems}
    for t in texts:
        assert "(E)" not in t and "(M)" not in t and "(H)" not in t
        assert "(nc-150+)" not in t and "(company question)" not in t
        assert "(lc-only)" not in t


def test_completion_parser_strips_difficulty_tag_from_canonical() -> None:
    """A checked line in today.md carries `(E) (Day 1)`-style annotations that the
    parser must strip to produce a ledger key matching the curriculum."""
    md = "- [x] [Arrays & Hashing] -> Contains Duplicate (E) (Day 1) ✅ 2026-05-11"
    assert parse_completions(md) == [
        Touch("[Arrays & Hashing] -> Contains Duplicate", date(2026, 5, 11))
    ]


def test_completion_parser_strips_source_tag_from_canonical() -> None:
    md = "- [x] [Segment Tree] -> Count of Smaller After Self (H) (nc-150+) (Day 54) ✅ 2026-07-03"
    assert parse_completions(md) == [
        Touch("[Segment Tree] -> Count of Smaller After Self", date(2026, 7, 3))
    ]


def test_completion_parser_strips_variant_of_tag_from_canonical() -> None:
    """`(variant of: X)` is human-readable lineage info; ledger keys must
    match the same problem text whether the tag is present or not."""
    md = "- [x] [1-D DP] -> House Robber II (M) (variant of: House Robber) (Day 31) ✅ 2026-06-10"
    assert parse_completions(md) == [
        Touch("[1-D DP] -> House Robber II", date(2026, 6, 10))
    ]


def test_curriculum_parser_canonical_text_omits_variant_of_tag() -> None:
    md = (
        "## DSA\n\n### Phase 4 — DP (4 new/day)\n\n"
        "#### 1-D DP\n\n- [ ] House Robber II (M) (variant of: House Robber)\n"
    )
    problems = parse_curriculum(md)
    assert len(problems) == 1
    assert problems[0].text == "[1-D DP] -> House Robber II"


# ─── compute_new source-tier ordering ─────────────────────────────────────────


def test_compute_new_prefers_nc_150_over_nc_150_plus_in_same_day() -> None:
    """The engine surfaces NC150 problems before non-NC150 patterns regardless
    of document position within a day."""
    curriculum = [
        Problem("[X] beyond-1", difficulty="M", source="nc-150+"),
        Problem("[X] core-nc-1", difficulty="M", source="nc-150"),
        Problem("[X] beyond-2", difficulty="H", source="nc-150+"),
        Problem("[X] core-nc-2", difficulty="M", source="nc-150"),
    ]
    new = compute_new(curriculum, ledger=[], limit=2)
    assert [p.text for p in new] == ["[X] core-nc-1", "[X] core-nc-2"]


def test_compute_new_falls_through_to_nc_150_plus_then_company_when_nc_150_is_drained() -> None:
    curriculum = [
        Problem("[X] nc-only", difficulty="M", source="nc-150"),
        Problem("[X] beyond-thing", difficulty="M", source="nc-150+"),
        Problem("[X] company-thing", difficulty="H", source="company question"),
    ]
    ledger = [Touch("[X] nc-only", date(2026, 7, 3))]
    new = compute_new(curriculum, ledger, limit=2)
    assert [p.text for p in new] == ["[X] beyond-thing", "[X] company-thing"]


def test_compute_new_within_a_source_tier_preserves_document_order() -> None:
    curriculum = [
        Problem("[X] P3", difficulty="M", source="nc-150"),
        Problem("[X] P1", difficulty="M", source="nc-150"),
        Problem("[X] P2", difficulty="M", source="nc-150"),
    ]
    new = compute_new(curriculum, ledger=[], limit=3)
    assert [p.text for p in new] == ["[X] P3", "[X] P1", "[X] P2"]


# ─── Difficulty surfaced in renderer ───────────────────────────────────────────


def test_recall_item_carries_difficulty_when_curriculum_is_provided() -> None:
    curriculum = [
        Problem("[A] Two Sum", difficulty="E", source="nc-150"),
    ]
    ledger = [Touch("[A] Two Sum", date(2026, 5, 1))]
    recall = compute_recall(ledger, today=date(2026, 5, 7), limit=10, curriculum=curriculum)
    assert recall[0].difficulty == "E"


def test_renderer_displays_difficulty_in_recall_lines() -> None:
    from recall_engine import RecallItem

    out = render_today(
        today=date(2026, 5, 7),
        recall=[
            RecallItem(
                problem="[A] Two Sum",
                touches=1,
                last_touched=date(2026, 5, 1),
                days_overdue=5,
                difficulty="E",
            )
        ],
        new=[],
    )
    # Difficulty is visible somewhere on the recall line
    recall_section = out.split("## New")[0]
    assert "(E)" in recall_section


def test_renderer_displays_difficulty_in_new_lines() -> None:
    out = render_today(
        today=date(2026, 5, 11),
        recall=[],
        new=[Problem("[A] Two Sum", difficulty="E", source="nc-150")],
    )
    new_section = out.split("## New")[1]
    assert "(E)" in new_section


def test_renderer_recall_line_hyperlinks_problem_name_when_link_available() -> None:
    """Recall lines mirror New lines: pattern as a wikilink and problem name as
    a markdown hyperlink so the NeetCode URL is clickable in Obsidian."""
    out = render_today(
        today=date(2026, 5, 7),
        recall=[
            RecallItem(
                problem="[Arrays & Hashing] -> Contains Duplicate",
                touches=1,
                last_touched=date(2026, 5, 1),
                days_overdue=5,
                difficulty="E",
                link="https://neetcode.io/problems/duplicate-integer",
            )
        ],
        new=[],
    )
    recall_section = out.split("## New")[0]
    assert (
        "[Contains Duplicate](https://neetcode.io/problems/duplicate-integer)"
        in recall_section
    )
    assert "[[arrays-hashing|Arrays & Hashing]]" in recall_section


def test_renderer_maintenance_line_hyperlinks_problem_name_when_link_available() -> None:
    out = render_today(
        today=date(2026, 5, 11),
        recall=[],
        new=[],
        maintenance=[
            RecallItem(
                problem="[Trees] -> Invert Binary Tree",
                touches=4,
                last_touched=date(2026, 5, 1),
                days_overdue=-10,
                difficulty="E",
                link="https://neetcode.io/problems/invert-a-binary-tree",
            )
        ],
    )
    maintenance_section = out.split("## Maintenance")[1]
    assert (
        "[Invert Binary Tree](https://neetcode.io/problems/invert-a-binary-tree)"
        in maintenance_section
    )
    assert "[[trees|Trees]]" in maintenance_section


def test_renderer_recall_line_falls_back_to_raw_text_when_no_link() -> None:
    """Without a link, no markdown hyperlink is emitted. The renderer also
    tolerates legacy/test fixtures that lack the canonical `[Pattern] -> Name`
    shape by passing the text through verbatim."""
    out = render_today(
        today=date(2026, 5, 7),
        recall=[
            RecallItem(
                problem="[A] Two Sum",
                touches=1,
                last_touched=date(2026, 5, 1),
                days_overdue=5,
                difficulty="E",
            )
        ],
        new=[],
    )
    recall_section = out.split("## New")[0]
    assert "](http" not in recall_section
    assert "[A] Two Sum" in recall_section


# ─── Saturday "this week's hardest" section ────────────────────────────────────


def test_renderer_includes_this_weeks_hardest_section_on_saturdays() -> None:
    """Saturday is reinforcement day — render an empty checklist where the user
    writes 2-3 problems they found hardest from this week's daily-hardest notes."""
    saturday = date(2026, 5, 16)
    assert saturday.weekday() == 5
    out = render_today(today=saturday, recall=[], new=[])
    assert "This week's hardest" in out


def test_renderer_omits_this_weeks_hardest_section_on_weekdays() -> None:
    monday = date(2026, 5, 11)
    assert monday.weekday() == 0
    out = render_today(today=monday, recall=[], new=[])
    assert "This week's hardest" not in out


def test_saturday_hardest_section_renders_empty_checkboxes_for_user_to_fill_in() -> None:
    """User writes problem names into the empty boxes; ticking them logs touches.
    The empty boxes themselves must be skipped by the completion parser."""
    saturday = date(2026, 5, 16)
    out = render_today(today=saturday, recall=[], new=[])
    # Section has writable bullet lines for the user
    section = out.split("This week's hardest")[1]
    assert section.count("- [ ]") >= 2


def test_completion_parser_ignores_empty_writable_checkboxes_on_saturday(tmp_path: Path) -> None:
    """The Saturday `- [ ]` empty bullets should never produce ledger entries
    even after Tasks plugin auto-stamps them when the user ticks them blank."""
    md = (
        "## This week's hardest — your pick\n\n"
        "- [x]  ✅ 2026-05-16\n"
        "- [x]  (just whitespace) ✅ 2026-05-16\n"
    )
    assert parse_completions(md) == []


# ─── Sprint day + phase math ──────────────────────────────────────────────────


def test_start_date_returns_none_for_empty_ledger() -> None:
    assert start_date([]) is None


def test_start_date_is_the_earliest_touch_in_the_ledger() -> None:
    """A friend running their own prep with a different start date will get
    Day 1 anchored to whenever they first checked something off."""
    ledger = [
        Touch("[A] Bar", date(2026, 5, 13)),
        Touch("[A] Foo", date(2026, 5, 11)),  # earliest
        Touch("[A] Baz", date(2026, 5, 14)),
    ]
    assert start_date(ledger) == date(2026, 5, 11)


def test_day_n_for_treats_start_date_as_day_one() -> None:
    start = date(2026, 5, 11)
    assert day_n_for(date(2026, 5, 11), start) == 1
    assert day_n_for(date(2026, 5, 12), start) == 2
    assert day_n_for(date(2026, 5, 18), start) == 8


# ─── Header rendering with phase + day ────────────────────────────────────────


def test_renderer_header_says_pre_prep_when_no_day_n() -> None:
    out = render_today(today=date(2026, 5, 8), recall=[], new=[])
    first_line = out.splitlines()[0]
    assert "Pre-prep" in first_line


def test_renderer_header_includes_day_n_when_provided() -> None:
    phase = Phase(number=1, name="Linear Patterns E+M", new_per_day=5)
    out = render_today(
        today=date(2026, 5, 11), recall=[], new=[], day_n=1, phase=phase
    )
    first_line = out.splitlines()[0]
    assert "Day 1" in first_line


def test_renderer_header_includes_phase_name_and_budget() -> None:
    """Phase header lists name + new/day budget so the reader sees the cap."""
    phase = Phase(number=1, name="Linear Patterns E+M", new_per_day=5)
    out = render_today(
        today=date(2026, 5, 11), recall=[], new=[], day_n=1, phase=phase
    )
    first_line = out.splitlines()[0]
    assert "Phase 1" in first_line
    assert "Linear Patterns E+M" in first_line
    assert "5 new/day" in first_line


def test_renderer_header_includes_phase_progress_when_total_phases_provided() -> None:
    """`Phase 2/7` shows current phase out of total at a glance."""
    phase = Phase(number=2, name="Linked List + Trees", new_per_day=4)
    out = render_today(
        today=date(2026, 5, 11),
        recall=[],
        new=[],
        day_n=1,
        phase=phase,
        total_phases=7,
    )
    first_line = out.splitlines()[0]
    assert "Phase 2/7" in first_line


def test_renderer_header_falls_back_to_day_only_when_phase_is_none() -> None:
    """If no phases are loaded, render the day number without phase metadata."""
    out = render_today(
        today=date(2026, 8, 15), recall=[], new=[], day_n=95, phase=None
    )
    first_line = out.splitlines()[0]
    assert "Day 95" in first_line
    assert "Phase" not in first_line


# ─── Pace projection ──────────────────────────────────────────────────────────


def test_avg_new_per_day_returns_none_for_empty_ledger() -> None:
    assert avg_new_per_day([], today=date(2026, 5, 11)) is None


def test_avg_new_per_day_is_distinct_problems_over_days_elapsed() -> None:
    """Day 1 with 3 distinct touches → 3.0/day. Day 2 with 5 distinct → 2.5/day."""
    ledger = [
        Touch("[A] Foo", date(2026, 5, 11)),
        Touch("[A] Bar", date(2026, 5, 11)),
        Touch("[A] Baz", date(2026, 5, 11)),
    ]
    assert avg_new_per_day(ledger, today=date(2026, 5, 11)) == 3.0

    ledger.extend(
        [
            Touch("[A] Qux", date(2026, 5, 12)),
            Touch("[A] Quux", date(2026, 5, 12)),
        ]
    )
    assert avg_new_per_day(ledger, today=date(2026, 5, 12)) == 2.5


def test_avg_new_per_day_counts_distinct_problems_not_touch_events() -> None:
    """Resolving the same problem twice does not inflate the acquisition rate."""
    ledger = [
        Touch("[A] Foo", date(2026, 5, 11)),
        Touch("[A] Foo", date(2026, 5, 12)),  # re-solve, not a new acquisition
        Touch("[A] Bar", date(2026, 5, 12)),
    ]
    # 2 distinct problems / 2 days elapsed = 1.0/day
    assert avg_new_per_day(ledger, today=date(2026, 5, 12)) == 1.0


def test_projected_end_date_returns_none_for_empty_ledger() -> None:
    curriculum = [Problem("[A] Foo", difficulty="E")]
    assert projected_end_date([], curriculum, today=date(2026, 5, 11)) is None


def test_projected_end_date_returns_today_when_curriculum_is_fully_touched() -> None:
    curriculum = [Problem("[A] Foo", difficulty="E")]
    ledger = [Touch("[A] Foo", date(2026, 5, 11))]
    assert projected_end_date(ledger, curriculum, today=date(2026, 5, 11)) == date(
        2026, 5, 11
    )


def test_projected_end_date_extrapolates_remaining_problems_at_current_pace() -> None:
    """Day 1, solved 3 of 9 → rate 3.0/day, 6 untouched → +2 days = May 13."""
    curriculum = [
        Problem(f"[A] P{i}", difficulty="E") for i in range(9)
    ]
    ledger = [
        Touch("[A] P0", date(2026, 5, 11)),
        Touch("[A] P1", date(2026, 5, 11)),
        Touch("[A] P2", date(2026, 5, 11)),
    ]
    assert projected_end_date(
        ledger, curriculum, today=date(2026, 5, 11)
    ) == date(2026, 5, 13)


def test_projected_end_date_self_corrects_as_pace_data_accrues() -> None:
    """Pace projections in early days swing wildly — that's OK as long as the
    function honestly reflects what the ledger says today."""
    curriculum = [
        Problem(f"[A] P{i}", difficulty="E") for i in range(20)
    ]
    fast_day_one = [Touch(f"[A] P{i}", date(2026, 5, 11)) for i in range(5)]
    end_after_fast_day = projected_end_date(
        fast_day_one, curriculum, today=date(2026, 5, 11)
    )
    # 5/day pace, 15 left → 3 days
    assert end_after_fast_day == date(2026, 5, 14)

    slow_day_two = fast_day_one + [Touch("[A] P5", date(2026, 5, 12))]
    end_after_slow_day = projected_end_date(
        slow_day_two, curriculum, today=date(2026, 5, 12)
    )
    # 6 distinct over 2 days = 3.0/day, 14 left → ceil(14/3) = 5 days
    assert end_after_slow_day == date(2026, 5, 17)


def test_renderer_omits_projection_line_when_no_projection_provided() -> None:
    out = render_today(today=date(2026, 5, 8), recall=[], new=[])
    assert "Projected" not in out


def test_renderer_includes_projection_line_below_header_when_data_present() -> None:
    out = render_today(
        today=date(2026, 5, 11),
        recall=[],
        new=[],
        day_n=1,
        projection=date(2026, 8, 15),
        projection_rate=3.0,
        projection_untouched=160,
    )
    lines = out.splitlines()
    assert "Day 1" in lines[0]
    # Projection sits on line 2 (zero-indexed: line index 2 after header + blank)
    assert "Projected acquisition complete" in lines[2]
    assert "Aug 15" in lines[2]
    assert "3.0 new/day" in lines[2]
    assert "160 left" in lines[2]


def test_renderer_omits_projection_line_when_curriculum_already_fully_touched() -> None:
    """Once everything is touched, the projection collapses to today — don't
    render a noisy '0 left' line; just suppress it."""
    out = render_today(
        today=date(2026, 8, 15),
        recall=[],
        new=[],
        day_n=97,
        projection=date(2026, 8, 15),
        projection_rate=2.0,
        projection_untouched=0,
    )
    assert "Projected" not in out


# ─── Coverage view (by-pattern) ───────────────────────────────────────────────


def test_curriculum_parser_captures_variant_of_relationship() -> None:
    md = (
        "## DSA\n\n### Phase 4 — DP (4 new/day)\n\n#### 1-D DP\n\n"
        "- [ ] House Robber (M)\n"
        "- [ ] House Robber II (M) (variant of: House Robber)\n"
    )
    problems = parse_curriculum(md)
    assert problems[0].variant_of is None
    assert problems[1].variant_of == "House Robber"


def test_problem_pattern_and_name_split_on_arrow() -> None:
    p = Problem("[Arrays & Hashing] -> Two Sum", difficulty="E")
    assert p.pattern == "Arrays & Hashing"
    assert p.name == "Two Sum"


# ─── Mock tracking ────────────────────────────────────────────────────────────


def test_parse_mocks_returns_empty_list_when_section_missing() -> None:
    assert parse_mocks("# No mocks section here") == []


def test_parse_mocks_extracts_pending_scheduled_completed_states() -> None:
    md = (
        "## Mocks\n\n"
        "- [ ] [m1] _pending_\n"
        "- [ ] [m2] Pramp · Trees · 📅 2026-05-20\n"
        "- [x] [m3] ✅ 2026-05-08 · note: weak on memo\n"
    )
    mocks = parse_mocks(md)
    assert mocks[0].status == "pending"
    assert mocks[1].scheduled_date == date(2026, 5, 20)
    assert mocks[1].platform == "Pramp"
    assert mocks[2].completed_date == date(2026, 5, 8)
    assert mocks[2].notes == "weak on memo"


def test_recompute_renders_next_mock_in_today_md(tmp_path: Path) -> None:
    """today.md gets a single 'Next mock' block between Progress and Recall.
    Mocks live in curriculum.md."""
    daily_md = tmp_path / "curriculum.md"
    daily_md.write_text(
        "## DSA\n\n### Phase 1 — Linear (5 new/day)\n\n"
        "#### A\n\n- [ ] Foo (E)\n\n"
        "## Mocks\n\n"
        "- [ ] [m1] Pramp · Trees · 📅 2026-05-13\n"
    )
    today_md = tmp_path / "today.md"
    ledger = tmp_path / "completions.jsonl"
    recompute(daily_md, today_md, ledger, today=date(2026, 5, 11))
    today_text = today_md.read_text()
    assert "## Next mock" in today_text
    progress_pos = today_text.find("## Progress")
    next_pos = today_text.find("## Next mock")
    recall_pos = today_text.find("## Recall")
    assert 0 <= progress_pos < next_pos < recall_pos


# ─── System Design chapter tracking ───────────────────────────────────────────


def test_parse_sd_chapters_returns_empty_list_when_section_missing() -> None:
    assert parse_sd_chapters("# No SD section") == []


def test_parse_sd_chapters_extracts_pending_and_completed_states() -> None:
    md = (
        "## System Design\n\n"
        "- [x] [ch-1] Alex Xu Vol 1 · Scale from Zero to Millions ✅ 2026-05-12\n"
        "- [ ] [ch-2] Alex Xu Vol 1 · Back-of-envelope estimation\n"
    )
    chapters = parse_sd_chapters(md)
    assert chapters[0].status == "completed"
    assert chapters[0].completed_date == date(2026, 5, 12)
    assert chapters[0].title == "Scale from Zero to Millions"
    assert chapters[0].book == "Alex Xu Vol 1"
    assert chapters[1].status == "pending"
    assert chapters[1].completed_date is None


def test_next_sd_chapter_returns_first_pending_in_document_order() -> None:
    chapters = [
        SDChapter(
            id="ch-1",
            title="Ch 1",
            book="Alex Xu Vol 1",
            status="completed",
            completed_date=date(2026, 5, 11),
        ),
        SDChapter(id="ch-2", title="Ch 2", book="Alex Xu Vol 1", status="pending"),
        SDChapter(id="ch-3", title="Ch 3", book="Alex Xu Vol 1", status="pending"),
    ]
    nxt = next_sd_chapter(chapters)
    assert nxt is not None and nxt.id == "ch-2"


def test_next_sd_chapter_returns_none_when_all_complete() -> None:
    chapters = [
        SDChapter(
            id="ch-1",
            title="Ch 1",
            book="Alex Xu Vol 1",
            status="completed",
            completed_date=date(2026, 5, 11),
        ),
    ]
    assert next_sd_chapter(chapters) is None


def test_render_today_surfaces_next_pending_sd_chapter_until_checked() -> None:
    nxt = SDChapter(
        id="ch-1",
        title="Scale from Zero to Millions",
        book="Alex Xu Vol 1",
        status="pending",
    )
    out = render_today(today=date(2026, 5, 11), recall=[], new=[], sd_next=nxt)
    assert "## Today's SD reading" in out
    assert "Alex Xu Vol 1" in out
    assert "Scale from Zero to Millions" in out


def test_render_today_omits_sd_section_when_no_pending_chapters() -> None:
    out = render_today(today=date(2026, 5, 11), recall=[], new=[], sd_next=None)
    assert "Today's SD reading" not in out


def test_recompute_reads_sd_chapters_and_renders_in_both_views(
    tmp_path: Path,
) -> None:
    daily_md = tmp_path / "curriculum.md"
    daily_md.write_text(
        "## DSA\n\n### Phase 1 — Linear (5 new/day)\n\n"
        "#### A\n\n- [ ] Foo (E)\n\n"
        "## System Design\n\n"
        "- [ ] [ch-1] Alex Xu Vol 1 · Scale from Zero to Millions\n"
    )
    today_md = tmp_path / "today.md"
    ledger = tmp_path / "completions.jsonl"
    recompute(daily_md, today_md, ledger, today=date(2026, 5, 11))
    assert "## Today's SD reading" in today_md.read_text()


# ─── Application-readiness gates ──────────────────────────────────────────────


def _em(text: str) -> Problem:
    return Problem(text, difficulty="M")


def _h(text: str) -> Problem:
    return Problem(text, difficulty="H")


def test_readiness_category_progress_reports_done_over_total() -> None:
    curriculum = [_em("[A] x"), _em("[A] y")]
    ledger = [Touch("[A] x", date(2026, 5, 11))]
    r = compute_readiness(curriculum, ledger, sd_chapters=[], mocks=[])
    assert r.em.done == 1 and r.em.total == 2
    assert r.em.fraction == 0.5


def test_render_readiness_block_shows_three_category_bars() -> None:
    em = CategoryProgress(name="E+M problems", done=4, total=8)
    sd = CategoryProgress(name="System Design", done=2, total=10)
    mocks = CategoryProgress(name="Mocks", done=1, total=3)
    out = "\n".join(render_readiness_block(Readiness(em, sd, mocks)))
    assert "## Progress" in out
    assert "E+M problems" in out and "50%" in out and "(4/8)" in out
    assert "System Design" in out and "20%" in out
    assert "Mocks" in out
    # No tier labels — the engine doesn't gate applications.
    assert "Fallback-ready" not in out
    assert "Target-ready" not in out
    assert "Stretch-ready" not in out


def test_render_today_renders_progress_block_above_recall_section() -> None:
    """Progress bars sit at the top so the user sees current state before scrolling."""
    em = CategoryProgress(name="E+M problems", done=0, total=1)
    sd = CategoryProgress(name="System Design", done=0, total=1)
    mocks = CategoryProgress(name="Mocks", done=0, total=1)
    readiness = Readiness(em=em, sd=sd, mocks=mocks)
    out = render_today(
        today=date(2026, 5, 11), recall=[], new=[], readiness=readiness
    )
    progress_pos = out.find("## Progress")
    recall_pos = out.find("## Recall")
    assert 0 <= progress_pos < recall_pos


# ─── Coverage.md mocks subsections (folded in from old mocks.md) ──────────────


# ─── Behavioral tracking ──────────────────────────────────────────────────────


def test_parse_behavioral_returns_empty_list_when_section_missing() -> None:
    assert parse_behavioral("# No behavioral section") == []


def test_parse_behavioral_extracts_pending_and_completed_entries() -> None:
    md = (
        "## Behavioral\n\n"
        "- [ ] [b1] Tell me about yourself\n"
        "- [x] [b2] Conflict story ✅ 2026-05-12 · note: use Datadog migration\n"
    )
    topics = parse_behavioral(md)
    assert topics[0].status == "pending"
    assert topics[1].completed_date == date(2026, 5, 12)
    assert topics[1].notes == "use Datadog migration"


def test_recompute_reads_behavioral_section_without_failing(tmp_path: Path) -> None:
    """Behavioral parsing during recompute is non-fatal even when no UI surfaces it."""
    daily_md = tmp_path / "curriculum.md"
    daily_md.write_text(
        "## DSA\n\n### Phase 1 — Linear (5 new/day)\n\n"
        "#### A\n\n- [ ] Foo (E)\n\n"
        "## Behavioral\n\n- [ ] [b1] Tell me about yourself\n"
    )
    today_md = tmp_path / "today.md"
    ledger = tmp_path / "completions.jsonl"
    recompute(daily_md, today_md, ledger, today=date(2026, 5, 11))
    assert today_md.exists()


# ─── Mock prerequisites ───────────────────────────────────────────────────────


def test_parse_mocks_extracts_em_and_sd_count_prereqs() -> None:
    md = (
        "## Mocks\n\n"
        "- [ ] [m1] _pending_ · prereq: 30 E+M, 5 SD\n"
    )
    mocks = parse_mocks(md)
    assert mocks[0].prerequisites == MockPrereq(em_problems=30, sd_chapters=5)


def test_parse_mocks_treats_missing_prereqs_as_none() -> None:
    """A mock without an inline `prereq:` segment parses to prerequisites=None."""
    md = "## Mocks\n\n- [ ] [m1] _pending_\n"
    mocks = parse_mocks(md)
    assert mocks[0].prerequisites is None


def test_mock_prereq_status_marks_each_dimension_as_met_or_unmet() -> None:
    mock = Mock(
        id="m1",
        status="pending",
        prerequisites=MockPrereq(em_problems=20, sd_chapters=5),
    )
    rows = mock_prereq_status(mock, em_done=24, sd_done=2)
    # First dim met (24 >= 20), second unmet (2 < 5)
    assert rows[0].label == "E/M problems"
    assert rows[0].met is True
    assert rows[0].current == 24
    assert rows[0].threshold == 20
    assert rows[1].label == "SD chapters"
    assert rows[1].met is False
    assert rows[1].current == 2
    assert rows[1].threshold == 5


def test_mock_prereq_status_skips_dimensions_with_zero_threshold() -> None:
    """If a mock only cares about E/M (no SD threshold), the SD row is omitted."""
    mock = Mock(
        id="m1",
        status="pending",
        prerequisites=MockPrereq(em_problems=20, sd_chapters=0),
    )
    rows = mock_prereq_status(mock, em_done=25, sd_done=0)
    assert len(rows) == 1
    assert rows[0].label == "E/M problems"


def test_mock_prereq_status_returns_empty_when_no_prereqs_defined() -> None:
    mock = Mock(id="m1", status="pending", prerequisites=None)
    assert mock_prereq_status(mock, em_done=99, sd_done=99) == []


def test_render_today_next_mock_block_shows_prereq_subbullet_when_defined() -> None:
    nxt = Mock(
        id="m1",
        status="scheduled",
        platform="Pramp",
        topic="Trees",
        scheduled_date=date(2026, 5, 20),
        prerequisites=MockPrereq(em_problems=25, sd_chapters=4),
    )
    out = render_today(
        today=date(2026, 5, 11),
        recall=[],
        new=[],
        next_up_mock=nxt,
        em_done=24,
        sd_done=5,
    )
    assert "## Next mock" in out
    assert "Prereqs:" in out
    assert "❌ 24/25 E/M problems" in out
    assert "✓ 5/4 SD chapters" in out


def test_parse_mocks_extracts_sd_chapter_ids_into_prerequisites_tuple() -> None:
    md = (
        "## Mocks\n\n"
        "- [ ] [m1] _pending_ · prereq: axu1-4, axu1-5, axu1-6\n"
    )
    mocks = parse_mocks(md)
    assert mocks[0].prerequisites == MockPrereq(
        em_problems=0,
        sd_chapters=0,
        sd_chapter_ids=("axu1-4", "axu1-5", "axu1-6"),
    )


def test_mock_prereq_status_with_chapter_ids_uses_chapter_completion_state(
    tmp_path: Path,
) -> None:
    """When `sd_chapter_ids` pins specific chapters, the SD row's `current`
    counts only those required chapters that are completed — not the global
    `sd_done` total — and the `detail` lists chapter titles inline with ✓."""
    chapters = [
        SDChapter(id="axu1-4", title="Ch 4 — Rate Limiter", book="Alex Xu Vol 1",
                  status="completed", completed_date=date(2026, 5, 1)),
        SDChapter(id="axu1-5", title="Ch 5 — Consistent Hashing",
                  book="Alex Xu Vol 1", status="pending"),
        SDChapter(id="axu1-6", title="Ch 6 — Key-Value Store",
                  book="Alex Xu Vol 1", status="pending"),
        SDChapter(id="axu1-7", title="Ch 7 — Unique ID", book="Alex Xu Vol 1",
                  status="completed", completed_date=date(2026, 5, 2)),
    ]
    mock = Mock(
        id="m1",
        status="pending",
        prerequisites=MockPrereq(sd_chapter_ids=("axu1-4", "axu1-5", "axu1-6")),
    )
    rows = mock_prereq_status(mock, em_done=0, sd_done=2, sd_chapters=chapters)
    assert len(rows) == 1
    assert rows[0].label == "SD chapters"
    assert rows[0].current == 1  # only axu1-4 is in the required set AND complete
    assert rows[0].threshold == 3
    assert rows[0].met is False
    assert rows[0].detail is not None
    assert "Ch 4 — Rate Limiter ✓" in rows[0].detail
    assert "Ch 5 — Consistent Hashing" in rows[0].detail
    assert "Ch 6 — Key-Value Store" in rows[0].detail


def test_render_today_chapter_id_prereq_lists_chapter_titles_inline() -> None:
    """Display: `❌ 1/3 SD chapters: Ch 4 — Rate Limiter ✓, Ch 5 — …, Ch 6 — …`"""
    chapters = [
        SDChapter(id="axu1-4", title="Ch 4 — Rate Limiter", book="Alex Xu Vol 1",
                  status="completed", completed_date=date(2026, 5, 1)),
        SDChapter(id="axu1-5", title="Ch 5 — Consistent Hashing",
                  book="Alex Xu Vol 1", status="pending"),
        SDChapter(id="axu1-6", title="Ch 6 — Key-Value Store",
                  book="Alex Xu Vol 1", status="pending"),
    ]
    nxt = Mock(
        id="m1",
        status="pending",
        platform="Pramp",
        topic="Trees",
        prerequisites=MockPrereq(sd_chapter_ids=("axu1-4", "axu1-5", "axu1-6")),
    )
    out = render_today(
        today=date(2026, 5, 11),
        recall=[],
        new=[],
        next_up_mock=nxt,
        em_done=0,
        sd_done=1,
        sd_chapters=chapters,
    )
    assert "❌ 1/3 SD chapters: Ch 4 — Rate Limiter ✓, " in out
    assert "Ch 5 — Consistent Hashing" in out
    assert "Ch 6 — Key-Value Store" in out


def test_write_curriculum_mocks_round_trips_sd_chapter_ids() -> None:
    """write_curriculum_mocks → parse_mocks must preserve `sd_chapter_ids`."""
    md = "## Mocks\n\n- [ ] [m1] _pending_\n"
    mocks_in = [
        Mock(
            id="m1",
            status="pending",
            prerequisites=MockPrereq(
                em_problems=15, sd_chapter_ids=("axu1-4", "axu1-5")
            ),
        )
    ]
    rewritten = write_curriculum_mocks(md, mocks_in)
    mocks_out = parse_mocks(rewritten)
    assert mocks_out[0].prerequisites is not None
    assert mocks_out[0].prerequisites.sd_chapter_ids == ("axu1-4", "axu1-5")
    assert mocks_out[0].prerequisites.em_problems == 15


# ─── Mock booking links ───────────────────────────────────────────────────────


def test_pending_mock_with_pramp_platform_renders_default_booking_link() -> None:
    """Pramp/Interviewing.io don't expose booking APIs, so we surface the
    platform's dashboard URL as a clickable link on the pending Next mock."""
    mock = Mock(id="m1", status="pending", platform="Pramp", topic="Trees")
    out = render_today(today=date(2026, 5, 11), recall=[], new=[], next_up_mock=mock)
    assert "Book: [Pramp](https://www.pramp.com/" in out


def test_pending_mock_with_explicit_booking_url_overrides_platform_default() -> None:
    mock = Mock(
        id="m1",
        status="pending",
        platform="Pramp",
        topic="Trees",
        booking_url="https://my-custom-booking.example/slot/abc123",
    )
    out = render_today(today=date(2026, 5, 11), recall=[], new=[], next_up_mock=mock)
    assert "https://my-custom-booking.example/slot/abc123" in out


def test_scheduled_mock_does_not_show_booking_link() -> None:
    """Already booked — no need to surface the booking URL."""
    mock = Mock(
        id="m1",
        status="scheduled",
        platform="Pramp",
        topic="Trees",
        scheduled_date=date(2026, 5, 20),
    )
    out = render_today(today=date(2026, 5, 11), recall=[], new=[], next_up_mock=mock)
    assert "Book:" not in out


# ─── today.md → curriculum.md mock wiring ────────────────────────────────────


def test_parse_mock_updates_extracts_scheduled_date_from_calendar_emoji() -> None:
    """User edits today.md to add 📅 after booking. Engine extracts the date
    on next recompute."""
    md = (
        "## Next mock\n\n"
        "- [ ] [mock-1] Pramp · Trees — _pending_ 📅 2026-05-20\n"
    )
    updates = parse_mock_updates(md, known_ids={"mock-1"})
    assert updates == [("mock-1", "scheduled", date(2026, 5, 20))]


def test_parse_mock_updates_extracts_completion_from_done_stamp() -> None:
    """Tasks plugin auto-stamps ✅ when the user checks the box. Engine treats
    that as the completion signal."""
    md = (
        "## Next mock\n\n"
        "- [x] [mock-1] Pramp · Trees · 📅 2026-05-20 ✅ 2026-05-22\n"
    )
    updates = parse_mock_updates(md, known_ids={"mock-1"})
    assert updates == [("mock-1", "completed", date(2026, 5, 22))]


def test_parse_mock_updates_ignores_unknown_ids() -> None:
    """Mock-id tag must match a known mock — protects against problem checkboxes
    or stray tags being parsed as mock state changes."""
    md = "- [ ] [random-tag] Whatever 📅 2026-05-20\n"
    assert parse_mock_updates(md, known_ids={"mock-1"}) == []


def test_parse_mock_updates_does_not_misread_problem_checkboxes() -> None:
    """DSA problem lines look like `[Pattern] -> Name` — the `->` distinguishes
    them from mock-id tags. parse_mock_updates only matches lines whose tag
    is in known_ids, so problems pass through untouched."""
    md = (
        "- [x] [Arrays & Hashing] -> Two Sum (E) ✅ 2026-05-20\n"
        "- [ ] [mock-1] Pramp · Trees 📅 2026-05-25\n"
    )
    updates = parse_mock_updates(md, known_ids={"mock-1"})
    assert updates == [("mock-1", "scheduled", date(2026, 5, 25))]


def test_apply_mock_updates_promotes_pending_to_scheduled_with_date() -> None:
    mocks = [Mock(id="m1", status="pending", platform="Pramp")]
    updates = [("m1", "scheduled", date(2026, 5, 20))]
    new_mocks, changes = apply_mock_updates(mocks, updates)
    assert changes == 1
    assert new_mocks[0].status == "scheduled"
    assert new_mocks[0].scheduled_date == date(2026, 5, 20)


def test_apply_mock_updates_promotes_scheduled_to_completed() -> None:
    mocks = [
        Mock(
            id="m1",
            status="scheduled",
            scheduled_date=date(2026, 5, 20),
        )
    ]
    new_mocks, changes = apply_mock_updates(
        mocks, [("m1", "completed", date(2026, 5, 22))]
    )
    assert changes == 1
    assert new_mocks[0].status == "completed"
    assert new_mocks[0].completed_date == date(2026, 5, 22)
    # Original scheduled_date is preserved on the completed mock
    assert new_mocks[0].scheduled_date == date(2026, 5, 20)


def test_apply_mock_updates_does_not_downgrade_completed_back_to_scheduled() -> None:
    """If a stale 📅 lingers on a completed mock's line, it shouldn't reset the
    completion. Completed is the terminal state."""
    mocks = [
        Mock(
            id="m1",
            status="completed",
            completed_date=date(2026, 5, 22),
        )
    ]
    new_mocks, changes = apply_mock_updates(
        mocks, [("m1", "scheduled", date(2026, 5, 20))]
    )
    assert changes == 0
    assert new_mocks[0].status == "completed"


def test_apply_mock_updates_returns_zero_changes_when_state_already_matches() -> None:
    mocks = [
        Mock(
            id="m1",
            status="scheduled",
            scheduled_date=date(2026, 5, 20),
        )
    ]
    _, changes = apply_mock_updates(
        mocks, [("m1", "scheduled", date(2026, 5, 20))]
    )
    assert changes == 0


def test_schedule_mock_writes_scheduled_date_into_curriculum() -> None:
    md = (
        "## Mocks\n\n"
        "- [ ] [mock-1] Pramp · Arrays & Hashing · _pending_\n"
    )
    new_md = schedule_mock(md, "mock-1", date(2026, 5, 17))
    mocks = parse_mocks(new_md)
    assert mocks[0].status == "scheduled"
    assert mocks[0].scheduled_date == date(2026, 5, 17)


def test_schedule_mock_can_reschedule_an_already_scheduled_mock() -> None:
    md = (
        "## Mocks\n\n"
        "- [ ] [mock-1] Pramp · Arrays & Hashing · 📅 2026-05-17\n"
    )
    new_md = schedule_mock(md, "mock-1", date(2026, 5, 24))
    assert parse_mocks(new_md)[0].scheduled_date == date(2026, 5, 24)


def test_schedule_mock_rejects_unknown_mock_id() -> None:
    md = "## Mocks\n\n- [ ] [mock-1] Pramp · _pending_\n"
    with pytest.raises(ValueError, match="unknown mock id"):
        schedule_mock(md, "mock-99", date(2026, 5, 17))


def test_schedule_mock_rejects_already_completed_mock() -> None:
    md = "## Mocks\n\n- [x] [mock-1] Pramp · ✅ 2026-05-08\n"
    with pytest.raises(ValueError, match="already completed"):
        schedule_mock(md, "mock-1", date(2026, 5, 24))


def test_complete_mock_marks_completion_date_and_preserves_status() -> None:
    md = (
        "## Mocks\n\n"
        "- [ ] [mock-1] Pramp · Arrays & Hashing · 📅 2026-05-17\n"
    )
    new_md = complete_mock(md, "mock-1", date(2026, 5, 17))
    mocks = parse_mocks(new_md)
    assert mocks[0].status == "completed"
    assert mocks[0].completed_date == date(2026, 5, 17)


def test_complete_mock_rejects_unknown_mock_id() -> None:
    md = "## Mocks\n\n- [ ] [mock-1] Pramp · _pending_\n"
    with pytest.raises(ValueError, match="unknown mock id"):
        complete_mock(md, "mock-99", date(2026, 5, 17))


# ─── Init / first-run seeding ─────────────────────────────────────────────────


def test_init_curriculum_file_seeds_when_target_missing(tmp_path: Path) -> None:
    template = tmp_path / "curriculum.template.md"
    template.write_text("# Curriculum\n\n- [ ] one\n")
    curriculum = tmp_path / "curriculum.md"

    wrote = init_curriculum_file(template, curriculum)

    assert wrote is True
    assert curriculum.read_text() == "# Curriculum\n\n- [ ] one\n"


def test_init_curriculum_file_is_a_noop_when_target_already_exists(tmp_path: Path) -> None:
    template = tmp_path / "curriculum.template.md"
    template.write_text("# pristine template\n")
    curriculum = tmp_path / "curriculum.md"
    curriculum.write_text("# my personal state — DO NOT CLOBBER\n- [x] something\n")

    wrote = init_curriculum_file(template, curriculum)

    assert wrote is False
    assert curriculum.read_text() == "# my personal state — DO NOT CLOBBER\n- [x] something\n"


def test_init_curriculum_file_overwrites_with_force(tmp_path: Path) -> None:
    template = tmp_path / "curriculum.template.md"
    template.write_text("# fresh\n")
    curriculum = tmp_path / "curriculum.md"
    curriculum.write_text("# stale state\n")

    wrote = init_curriculum_file(template, curriculum, force=True)

    assert wrote is True
    assert curriculum.read_text() == "# fresh\n"


def test_init_curriculum_file_raises_when_template_missing(tmp_path: Path) -> None:
    template = tmp_path / "missing.template.md"
    curriculum = tmp_path / "curriculum.md"

    with pytest.raises(FileNotFoundError):
        init_curriculum_file(template, curriculum)


def test_ensure_runtime_dirs_is_idempotent(tmp_path: Path) -> None:
    prep_data = tmp_path / "prep-data"
    problems = tmp_path / "problems"
    ensure_runtime_dirs(prep_data, problems)
    # Second call must not fail even though dirs already exist.
    ensure_runtime_dirs(prep_data, problems)
    assert prep_data.is_dir()
    assert problems.is_dir()


def test_write_curriculum_mocks_round_trips_through_parse_mocks() -> None:
    """write_curriculum_mocks → parse_mocks should preserve every modeled
    field including prereqs and booking_url."""
    md = "## Mocks\n\n- [ ] [m1] _pending_\n- [ ] [m2] _pending_\n"
    original = [
        Mock(
            id="m1",
            status="scheduled",
            platform="Pramp",
            topic="Trees",
            scheduled_date=date(2026, 5, 20),
            prerequisites=MockPrereq(em_problems=15, sd_chapters=2),
            booking_url="https://my-booking.example/abc",
        ),
        Mock(
            id="m2",
            status="completed",
            platform="Interviewing.io",
            topic="DP",
            completed_date=date(2026, 5, 8),
            notes="weak on memo",
        ),
    ]
    rewritten = write_curriculum_mocks(md, original)
    reloaded = parse_mocks(rewritten)
    assert reloaded == original


def test_recompute_folds_today_md_calendar_edit_into_curriculum_md(
    tmp_path: Path,
) -> None:
    """End-to-end: user adds 📅 to today.md, runs recompute, curriculum.md
    reflects the new scheduled state."""
    daily_md = tmp_path / "curriculum.md"
    daily_md.write_text(
        "## DSA\n\n### Phase 1 — Linear (5 new/day)\n\n"
        "#### A\n\n- [ ] Foo (E)\n\n"
        '## Mocks\n\n'
        '- [ ] [mock-1] Pramp · Trees · _pending_\n'
    )
    today_md = tmp_path / "today.md"
    today_md.write_text(
        "# Today\n\n"
        "## Next mock\n\n"
        "- [ ] [mock-1] Pramp · Trees — _pending_ 📅 2026-05-20\n"
    )
    ledger = tmp_path / "completions.jsonl"
    recompute(
        daily_md,
        today_md,
        ledger,
        today=date(2026, 5, 11),
    )
    updated = parse_mocks(daily_md.read_text())
    assert updated[0].status == "scheduled"
    assert updated[0].scheduled_date == date(2026, 5, 20)


def test_recompute_folds_today_md_completion_check_into_curriculum_md(
    tmp_path: Path,
) -> None:
    """Tasks plugin checks the box and auto-stamps ✅. Recompute folds that
    completion stamp back into curriculum.md."""
    daily_md = tmp_path / "curriculum.md"
    daily_md.write_text(
        "## DSA\n\n### Phase 1 — Linear (5 new/day)\n\n"
        "#### A\n\n- [ ] Foo (E)\n\n"
        '## Mocks\n\n'
        '- [ ] [mock-1] Pramp · Trees · 📅 2026-05-20\n'
    )
    today_md = tmp_path / "today.md"
    today_md.write_text(
        "# Today\n\n"
        "## Next mock\n\n"
        "- [x] [mock-1] Pramp · Trees · 📅 2026-05-20 ✅ 2026-05-22\n"
    )
    ledger = tmp_path / "completions.jsonl"
    recompute(
        daily_md,
        today_md,
        ledger,
        today=date(2026, 5, 23),
    )
    updated = parse_mocks(daily_md.read_text())
    assert updated[0].status == "completed"
    assert updated[0].completed_date == date(2026, 5, 22)


# ─── Phase model + phase-aware compute_new ───────────────────────────────────

PHASE_LINEAR = Phase(number=1, name="Linear Patterns E+M", new_per_day=5)
PHASE_TREES = Phase(number=2, name="Trees", new_per_day=4)
PHASE_HARD = Phase(number=3, name="Hard Problems", new_per_day=2)
PHASE_REINFORCE = Phase(number=8, name="Reinforcement", new_per_day=0)


def _p(
    text: str,
    diff: str = "M",
    source: str = "nc-150",
    phase: int | None = None,
) -> Problem:
    return Problem(text=text, difficulty=diff, source=source, phase=phase)  # type: ignore[arg-type]


def test_current_phase_picks_first_phase_with_untouched_problems() -> None:
    curriculum = [
        _p("[Arrays & Hashing] -> Two Sum", "E", phase=1),
        _p("[Trees] -> Invert Binary Tree", "E", phase=2),
    ]
    ledger: list[Touch] = []
    assert current_phase(curriculum, ledger, [PHASE_LINEAR, PHASE_TREES]).number == 1


def test_current_phase_advances_when_phase_patterns_drained() -> None:
    """All Phase 1 problems touched → current phase rolls to Phase 2."""
    curriculum = [
        _p("[Arrays & Hashing] -> Two Sum", "E", phase=1),
        _p("[Two Pointers] -> Valid Palindrome", "E", phase=1),
        _p("[Trees] -> Invert Binary Tree", "E", phase=2),
    ]
    ledger = [
        Touch("[Arrays & Hashing] -> Two Sum", date(2026, 5, 9)),
        Touch("[Two Pointers] -> Valid Palindrome", date(2026, 5, 9)),
    ]
    assert current_phase(curriculum, ledger, [PHASE_LINEAR, PHASE_TREES]).number == 2


def test_compute_new_filters_to_problems_assigned_to_current_phase() -> None:
    """Phase membership is encoded on Problem.phase; compute_new filters by it."""
    curriculum = [
        _p("[Arrays & Hashing] -> Two Sum", "E", phase=1),
        _p("[Trees] -> Invert Binary Tree", "E", phase=2),
        _p("[Two Pointers] -> 3Sum", "M", phase=1),
    ]
    new = compute_new(curriculum, ledger=[], phase=PHASE_LINEAR)
    assert all(p.phase == 1 for p in new)
    assert "[Trees] -> Invert Binary Tree" not in {p.text for p in new}


def test_compute_new_returns_empty_when_phase_new_per_day_is_zero() -> None:
    """Reinforcement phase: ledger is full, no acquisition wanted."""
    curriculum = [_p("[Arrays & Hashing] -> Two Sum", "E", phase=8)]
    assert compute_new(curriculum, ledger=[], phase=PHASE_REINFORCE) == []


def test_parse_phases_extracts_phase_metadata_from_curriculum_md() -> None:
    """Phase budgets live inline in `### Phase N — Name (X new/day)` headings."""
    md = (
        "## DSA\n\n"
        "### Phase 1 — Linear (5 new/day)\n\n"
        "#### A\n\n- [ ] X (E)\n\n"
        "### Phase 5 — Hards (2 new/day)\n\n"
        "#### B\n\n- [ ] Y (H)\n"
    )
    phases = parse_phases(md)
    assert phases == [
        Phase(number=1, name="Linear", new_per_day=5),
        Phase(number=5, name="Hards", new_per_day=2),
    ]


def test_parse_curriculum_assigns_phase_to_each_problem() -> None:
    md = (
        "## DSA\n\n"
        "### Phase 1 — Linear (5 new/day)\n\n"
        "#### Arrays & Hashing\n\n- [ ] Two Sum (E)\n\n"
        "### Phase 5 — Hards (2 new/day)\n\n"
        "#### Two Pointers\n\n- [ ] Trapping Rain Water (H)\n"
    )
    problems = parse_curriculum(md)
    by_text = {p.text: p for p in problems}
    assert by_text["[Arrays & Hashing] -> Two Sum"].phase == 1
    assert by_text["[Two Pointers] -> Trapping Rain Water"].phase == 5


def test_parse_curriculum_ignores_problems_outside_dsa_section() -> None:
    """Problems under `## System Design`, `## Mocks`, etc. are not DSA."""
    md = (
        "## DSA\n\n"
        "### Phase 1 — Linear (5 new/day)\n\n"
        "#### Arrays & Hashing\n\n- [ ] Two Sum (E)\n\n"
        "## System Design\n\n- [ ] axu1-1 · Alex Xu Vol 1 · Ch 1\n\n"
        "## Mocks\n\n- [ ] [mock-1] Pramp · Trees — pending\n"
    )
    problems = parse_curriculum(md)
    assert [p.text for p in problems] == ["[Arrays & Hashing] -> Two Sum"]


# ─── Time blocks in today.md ─────────────────────────────────────────────────


def test_time_blocks_show_default_schedule_on_non_mock_day() -> None:
    out = render_today(today=date(2026, 5, 11), recall=[], new=[], day_n=1, mock_today=False)
    assert "## Time blocks" in out
    assert "9:00–13:00  Recall" in out
    assert "14:00–15:30 System Design" in out
    assert "15:30–19:30 DSA New" in out
    assert "Mock" not in out  # no mock block on a non-mock day


def test_time_blocks_shift_sd_when_today_is_a_mock_day() -> None:
    """Mock day: 14:00–16:00 Mock pushes SD to 16:00–17:30 and DSA New to 17:30–19:30."""
    out = render_today(today=date(2026, 5, 12), recall=[], new=[], day_n=2, mock_today=True)
    blocks = out.split("## Time blocks", 1)[1]
    assert "14:00–16:00 Mock" in blocks
    assert "16:00–17:30 System Design" in blocks
    assert "17:30–19:30 DSA New" in blocks


# ─── DSA bidirectional sync (curriculum.md ↔ ledger) ─────────────────────────


def _dsa_md(*, two_sum_touch: date | None = None) -> str:
    """Compose a minimal DSA section. If `two_sum_touch` is set, render Two
    Sum as touched on that date with one ticked sub-bullet + 4 padding slots."""
    if two_sum_touch is None:
        return (
            "## DSA\n\n### Phase 1 — Linear (5 new/day)\n\n"
            "#### Arrays & Hashing\n\n- Two Sum (E) · 0/5\n"
        )
    return (
        "## DSA\n\n### Phase 1 — Linear (5 new/day)\n\n"
        "#### Arrays & Hashing\n\n"
        f"- Two Sum (E) · 1/5 (next due {(two_sum_touch + timedelta(days=1)).isoformat()})\n"
        f"  - [x] ✅ {two_sum_touch.isoformat()}\n"
        "  - [ ]\n"
        "  - [ ]\n"
        "  - [ ]\n"
        "  - [ ]\n"
    )


def test_parse_curriculum_dsa_state_returns_set_of_touch_dates_per_problem() -> None:
    md = (
        "## DSA\n\n### Phase 1 — Linear (5 new/day)\n\n"
        "#### Arrays & Hashing\n\n"
        "- Two Sum (E) · 2/5\n"
        "  - [x] ✅ 2026-05-11\n"
        "  - [x] ✅ 2026-05-14\n"
        "  - [ ]\n"
        "  - [ ]\n"
        "  - [ ]\n"
        "- 3Sum (M) · 0/5\n"
    )
    state = parse_curriculum_dsa_state(md)
    assert state == {
        "[Arrays & Hashing] -> Two Sum": {date(2026, 5, 11), date(2026, 5, 14)},
        "[Arrays & Hashing] -> 3Sum": set(),
    }


def test_parse_curriculum_dsa_state_skips_empty_padding_subbullets() -> None:
    md = (
        "## DSA\n\n### Phase 1 — Linear (5 new/day)\n\n"
        "#### Arrays & Hashing\n\n"
        "- Two Sum (E) · 0/5\n"
        "  - [ ]\n  - [ ]\n  - [ ]\n  - [ ]\n  - [ ]\n"
    )
    state = parse_curriculum_dsa_state(md)
    assert state == {"[Arrays & Hashing] -> Two Sum": set()}


def test_parse_curriculum_dsa_state_warns_on_malformed_subbullet_date(
    capsys: pytest.CaptureFixture[str],
) -> None:
    md = (
        "## DSA\n\n### Phase 1 — Linear (5 new/day)\n\n"
        "#### Arrays & Hashing\n\n"
        "- Two Sum (E) · 1/5\n"
        "  - [x] ✅ 2/2/26\n"
        "  - [x] ✅ 2026-05-11\n"
    )
    state = parse_curriculum_dsa_state(md)
    captured = capsys.readouterr()
    assert state == {"[Arrays & Hashing] -> Two Sum": {date(2026, 5, 11)}}
    assert "malformed" in captured.err.lower()
    assert "Two Sum" in captured.err


def test_parse_curriculum_dsa_state_treats_legacy_checked_parent_as_one_touch() -> None:
    """Migration: a legacy `- [x] Name (E) ✅ DATE` line (no sub-bullets yet)
    counts as a single touch on that date so the first recompute round-trip
    matches the ledger and won't purge."""
    md = (
        "## DSA\n\n### Phase 1 — Linear (5 new/day)\n\n"
        "#### Arrays & Hashing\n\n- [x] Two Sum (E) ✅ 2026-05-09\n"
    )
    assert parse_curriculum_dsa_state(md) == {
        "[Arrays & Hashing] -> Two Sum": {date(2026, 5, 9)}
    }


def test_recompute_logs_curriculum_md_dsa_subbullet_as_a_ledger_touch(
    tmp_path: Path,
) -> None:
    """User adds a sub-bullet `- [x] ✅ DATE` directly in curriculum.md →
    recompute appends a touch dated DATE (covers backdated hand-typing)."""
    daily_md = tmp_path / "curriculum.md"
    daily_md.write_text(_dsa_md(two_sum_touch=date(2026, 5, 11)))
    today_md = tmp_path / "today.md"
    ledger = tmp_path / "completions.jsonl"
    recompute(daily_md, today_md, ledger, today=date(2026, 5, 11))
    touches = [json.loads(l) for l in ledger.read_text().splitlines() if l]
    assert any(
        t["problem"] == "[Arrays & Hashing] -> Two Sum" and t["on"] == "2026-05-11"
        for t in touches
    )


def test_recompute_purges_specific_touch_when_subbullet_removed(
    tmp_path: Path,
) -> None:
    """User deletes one `- [x] ✅ DATE` sub-bullet → only that specific
    `(problem, date)` is removed from the ledger; other touches preserved."""
    daily_md = tmp_path / "curriculum.md"
    # Two ticks remaining (May 11 + May 14) — May 9 was deleted by the user.
    daily_md.write_text(
        "## DSA\n\n### Phase 1 — Linear (5 new/day)\n\n#### Arrays & Hashing\n\n"
        "- Two Sum (E) · 2/5\n"
        "  - [x] ✅ 2026-05-11\n"
        "  - [x] ✅ 2026-05-14\n"
        "  - [ ]\n  - [ ]\n  - [ ]\n"
    )
    today_md = tmp_path / "today.md"
    ledger = tmp_path / "completions.jsonl"
    ledger.write_text(
        '{"problem": "[Arrays & Hashing] -> Two Sum", "on": "2026-05-09"}\n'
        '{"problem": "[Arrays & Hashing] -> Two Sum", "on": "2026-05-11"}\n'
        '{"problem": "[Arrays & Hashing] -> Two Sum", "on": "2026-05-14"}\n'
    )
    recompute(daily_md, today_md, ledger, today=date(2026, 5, 14))
    touches = [json.loads(l) for l in ledger.read_text().splitlines() if l]
    on_dates = {t["on"] for t in touches}
    assert on_dates == {"2026-05-11", "2026-05-14"}


def test_recompute_full_purge_when_all_subbullets_removed(
    tmp_path: Path,
) -> None:
    """When the user removes every sub-bullet for a problem (counter 0/5)
    while OTHER problems still have ticks, treat as a destructive full
    purge: drop all of that problem's ledger entries and warn."""
    daily_md = tmp_path / "curriculum.md"
    daily_md.write_text(
        "## DSA\n\n### Phase 1 — Linear (5 new/day)\n\n#### Arrays & Hashing\n\n"
        "- Two Sum (E) · 1/5\n"
        "  - [x] ✅ 2026-05-11\n"
        "  - [ ]\n  - [ ]\n  - [ ]\n  - [ ]\n"
        "- Group Anagrams (M) · 0/5\n"
    )
    today_md = tmp_path / "today.md"
    ledger = tmp_path / "completions.jsonl"
    ledger.write_text(
        '{"problem": "[Arrays & Hashing] -> Two Sum", "on": "2026-05-11"}\n'
        '{"problem": "[Arrays & Hashing] -> Group Anagrams", "on": "2026-05-09"}\n'
    )
    recompute(daily_md, today_md, ledger, today=date(2026, 5, 14))
    remaining = ledger.read_text()
    assert "Group Anagrams" not in remaining
    assert "Two Sum" in remaining


def test_recompute_dry_run_does_not_purge_ledger(tmp_path: Path) -> None:
    """`dry_run=True` must not mutate the ledger when a curriculum.md
    sub-bullet uncheck would otherwise remove a touch."""
    daily_md = tmp_path / "curriculum.md"
    daily_md.write_text(
        "## DSA\n\n### Phase 1 — Linear (5 new/day)\n\n#### Arrays & Hashing\n\n"
        "- Two Sum (E) · 1/5\n"
        "  - [x] ✅ 2026-05-11\n"
        "  - [ ]\n  - [ ]\n  - [ ]\n  - [ ]\n"
        "- Group Anagrams (M) · 0/5\n"
    )
    today_md = tmp_path / "today.md"
    ledger = tmp_path / "completions.jsonl"
    original = (
        '{"problem": "[Arrays & Hashing] -> Two Sum", "on": "2026-05-11"}\n'
        '{"problem": "[Arrays & Hashing] -> Group Anagrams", "on": "2026-05-09"}\n'
    )
    ledger.write_text(original)
    recompute(daily_md, today_md, ledger, today=date(2026, 5, 14), dry_run=True)
    assert ledger.read_text() == original


def test_recompute_skips_purge_on_pristine_all_unchecked_curriculum_migration(
    tmp_path: Path,
) -> None:
    """Migration safeguard: when curriculum.md is freshly restructured (every
    problem at `0/N`, no sub-bullets) but the ledger has entries, treat as a
    migration. Don't purge — recompute's render pass re-syncs the ticks from
    the ledger. Input fixture uses the legacy `0/5` denominator to prove
    backwards compatibility; output is re-rendered under the current cap."""
    daily_md = tmp_path / "curriculum.md"
    daily_md.write_text(
        "## DSA\n\n### Phase 1 — Linear (5 new/day)\n\n#### Arrays & Hashing\n\n"
        "- Two Sum (E) · 0/5\n"
    )
    today_md = tmp_path / "today.md"
    ledger = tmp_path / "completions.jsonl"
    ledger.write_text(
        '{"problem": "[Arrays & Hashing] -> Two Sum", "on": "2026-05-09"}\n'
    )
    recompute(daily_md, today_md, ledger, today=date(2026, 5, 11))
    assert "Two Sum" in ledger.read_text()
    rendered = daily_md.read_text()
    # curriculum.md re-renders with sub-bullets reflecting the ledger.
    assert "- Two Sum (E) · 1/4" in rendered
    assert "  - [x] ✅ 2026-05-09" in rendered


def test_write_curriculum_dsa_pads_partial_progress_up_to_mastery_slot_count() -> None:
    md = (
        "## DSA\n\n### Phase 1 — Linear (5 new/day)\n\n"
        "#### Arrays & Hashing\n\n- Two Sum (E)\n"
    )
    ledger = [
        Touch("[Arrays & Hashing] -> Two Sum", date(2026, 5, 9)),
        Touch("[Arrays & Hashing] -> Two Sum", date(2026, 5, 11)),
    ]
    out = write_curriculum_dsa(md, ledger, today=date(2026, 5, 11))
    assert "- Two Sum (E) · 2/4 (next due 2026-05-14)" in out
    # Both ticks render in chronological order, plus 2 empty padding slots.
    assert "  - [x] ✅ 2026-05-09" in out
    assert "  - [x] ✅ 2026-05-11" in out
    assert out.count("  - [ ]") == 2


def test_write_curriculum_dsa_saturates_at_mastery_cap_with_no_padding() -> None:
    md = (
        "## DSA\n\n### Phase 1 — Linear (5 new/day)\n\n"
        "#### Arrays & Hashing\n\n- Two Sum (E)\n"
    )
    ledger = [
        Touch("[Arrays & Hashing] -> Two Sum", date(2026, m, d))
        for (m, d) in [(5, 9), (5, 12), (5, 19), (6, 9), (8, 8)]
    ]
    out = write_curriculum_dsa(md, ledger, today=date(2026, 8, 8))
    # 5 touches; counter saturates at 4/4 (mastery cap), all 5 ticks still render.
    assert "- Two Sum (E) · 4/4" in out
    assert "  - [ ]" not in out


def test_write_curriculum_dsa_renders_all_touches_past_mastery_cap_no_truncation() -> None:
    """Past the 4-touch mastery cap, the counter saturates at 4/4 but every
    touch still renders — full history is preserved, not truncated."""
    md = (
        "## DSA\n\n### Phase 1 — Linear (5 new/day)\n\n"
        "#### Arrays & Hashing\n\n- Two Sum (E)\n"
    )
    dates_ = [date(2026, 5, 9), date(2026, 5, 12), date(2026, 5, 19),
              date(2026, 6, 9), date(2026, 8, 8), date(2026, 10, 7)]
    ledger = [Touch("[Arrays & Hashing] -> Two Sum", d) for d in dates_]
    out = write_curriculum_dsa(md, ledger, today=date(2026, 10, 7))
    assert "- Two Sum (E) · 4/4" in out  # counter saturates at 4/4
    for d in dates_:
        assert f"  - [x] ✅ {d.isoformat()}" in out
    assert "  - [ ]" not in out


def test_write_curriculum_dsa_renders_no_subbullets_for_untouched_problem() -> None:
    md = (
        "## DSA\n\n### Phase 1 — Linear (5 new/day)\n\n"
        "#### Arrays & Hashing\n\n- Two Sum (E)\n"
    )
    out = write_curriculum_dsa(md, ledger=[], today=date(2026, 5, 11))
    assert "- Two Sum (E) · 0/4" in out
    assert "✅" not in out
    assert "  - [ ]" not in out  # untouched: no padding slots either
    assert "(next due" not in out
    assert "(overdue" not in out


def test_write_curriculum_dsa_renders_overdue_when_today_past_due_date() -> None:
    md = (
        "## DSA\n\n### Phase 1 — Linear (5 new/day)\n\n"
        "#### Arrays & Hashing\n\n- Two Sum (E)\n"
    )
    ledger = [Touch("[Arrays & Hashing] -> Two Sum", date(2026, 5, 11))]
    # 1 touch → next due May 12; today May 18 → overdue 6d.
    out = write_curriculum_dsa(md, ledger, today=date(2026, 5, 18))
    assert "- Two Sum (E) · 1/4 (overdue 6d)" in out


def test_write_curriculum_dsa_drops_legacy_checked_stamp_in_re_render() -> None:
    """Re-rendering from a ledger that has no entries for a problem strips
    the legacy `[x] ... ✅ DATE` form and produces the new `· 0/4` shape."""
    md = (
        "## DSA\n\n### Phase 1 — Linear (5 new/day)\n\n"
        "#### Arrays & Hashing\n\n- [x] Two Sum (E) ✅ 2026-05-08\n"
    )
    out = write_curriculum_dsa(md, ledger=[], today=date(2026, 5, 11))
    assert "- Two Sum (E) · 0/4" in out
    assert "✅" not in out


def test_recompute_writes_back_curriculum_md_to_match_ledger(tmp_path: Path) -> None:
    """After today.md ticks log a touch, curriculum.md's DSA problem becomes
    `· 1/4` with one ticked sub-bullet + 3 empty padding slots."""
    daily_md = tmp_path / "curriculum.md"
    daily_md.write_text(_dsa_md(two_sum_touch=None))
    today_md = tmp_path / "today.md"
    today_md.write_text("- [x] [Arrays & Hashing] -> Two Sum ✅ 2026-05-11\n")
    ledger = tmp_path / "completions.jsonl"
    recompute(daily_md, today_md, ledger, today=date(2026, 5, 11))
    text = daily_md.read_text()
    assert "- Two Sum (E) · 1/4 (next due 2026-05-12)" in text
    assert "  - [x] ✅ 2026-05-11" in text
    assert text.count("  - [ ]") == 3


# ─── Hardest-of-day pick tracking ──────────────────────────────────────────────


def test_parse_curriculum_captures_problem_link_from_markdown_link() -> None:
    md = (
        "## DSA\n\n### Phase 1 — Linear (5 new/day)\n\n"
        "#### Arrays & Hashing\n\n"
        "- [Two Sum](https://neetcode.io/problems/two-sum) (E) · 0/5\n"
    )
    problems = parse_curriculum(md)
    assert problems[0].link == "https://neetcode.io/problems/two-sum"
    assert problems[0].text == "[Arrays & Hashing] -> Two Sum"


def test_parse_curriculum_drops_relative_paths_for_company_stub_links() -> None:
    md = (
        "## DSA\n\n### Phase 1 — Linear (5 new/day)\n\n"
        "#### Arrays & Hashing\n\n"
        "- [Foo Stub](./problems/company/foo.md) (M) (company question) · 0/5\n"
    )
    assert parse_curriculum(md)[0].link is None


def test_parse_hardest_marks_extracts_ticks_under_dated_section() -> None:
    today_md = (
        "# Today — Mon May 11, 2026\n\n"
        "## Today's hardest (2026-05-11) — pick from today's New\n\n"
        "- [x] [Arrays & Hashing] -> Two Sum (E)\n"
        "- [ ] [Arrays & Hashing] -> Group Anagrams (M)\n"
    )
    marks = parse_hardest_marks(today_md, fallback_date=date(2026, 5, 12))
    assert marks == [HardestMark("[Arrays & Hashing] -> Two Sum", date(2026, 5, 11))]


def test_parse_hardest_marks_falls_back_to_supplied_date_when_heading_undated() -> None:
    today_md = (
        "## Today's hardest — pick from today's New\n\n"
        "- [x] [Heap] -> Task Scheduler (M)\n"
    )
    marks = parse_hardest_marks(today_md, fallback_date=date(2026, 5, 11))
    assert marks == [HardestMark("[Heap] -> Task Scheduler", date(2026, 5, 11))]


def test_parse_hardest_marks_ignores_ticks_outside_section() -> None:
    today_md = (
        "## Recall — most overdue first\n\n"
        "- [x] [Arrays & Hashing] -> Two Sum (E) ✅ 2026-05-11\n\n"
        "## Today's hardest (2026-05-11)\n\n"
        "- [x] [Heap] -> Task Scheduler (M)\n\n"
        "## New — next from the curriculum\n\n"
        "- [x] [Stack] -> Min Stack (E) ✅ 2026-05-11\n"
    )
    marks = parse_hardest_marks(today_md, fallback_date=date(2026, 5, 11))
    assert marks == [HardestMark("[Heap] -> Task Scheduler", date(2026, 5, 11))]


def test_parse_completions_skips_ticks_under_hardest_section() -> None:
    today_md = (
        "## Recall — most overdue first\n\n"
        "- [x] [Arrays & Hashing] -> Two Sum (E) ✅ 2026-05-11\n\n"
        "## Today's hardest (2026-05-11)\n\n"
        "- [x] [Heap] -> Task Scheduler (M) ✅ 2026-05-11\n"
    )
    touches = parse_completions(today_md)
    assert len(touches) == 1
    assert touches[0].problem == "[Arrays & Hashing] -> Two Sum"


def test_append_to_hardest_ledger_dedupes_existing_entries(tmp_path: Path) -> None:
    path = tmp_path / "hardest.jsonl"
    mark = HardestMark("[Heap] -> Task Scheduler", date(2026, 5, 11))
    assert append_to_hardest_ledger(path, [mark]) == 1
    # Re-applying the same mark is a no-op.
    assert append_to_hardest_ledger(path, [mark]) == 0
    assert load_hardest_ledger(path) == [mark]


def test_render_today_renders_hardest_pick_section_on_weekdays_with_links() -> None:
    new = [
        Problem(
            text="[Arrays & Hashing] -> Two Sum",
            difficulty="E",
            link="https://neetcode.io/problems/two-sum",
        ),
        Problem(
            text="[Arrays & Hashing] -> Group Anagrams",
            difficulty="M",
            link="https://neetcode.io/problems/group-anagrams",
        ),
    ]
    md = render_today(today=date(2026, 5, 11), recall=[], new=new)  # Monday
    assert "## Today's hardest (2026-05-11) — pick from today's New" in md
    assert (
        "- [ ] [[arrays-hashing|Arrays & Hashing]] -> [Two Sum](https://neetcode.io/problems/two-sum) (E)"
        in md
    )
    assert (
        "- [ ] [[arrays-hashing|Arrays & Hashing]] -> [Group Anagrams](https://neetcode.io/problems/group-anagrams) (M)"
        in md
    )


def test_render_today_skips_hardest_pick_section_on_saturday() -> None:
    new = [Problem(text="[Heap] -> Task Scheduler", difficulty="M")]
    md = render_today(today=date(2026, 5, 16), recall=[], new=new)  # Saturday
    assert "## Today's hardest" not in md
    assert "## This week's hardest" in md


def test_render_today_skips_hardest_pick_section_on_sunday() -> None:
    new = [Problem(text="[Heap] -> Task Scheduler", difficulty="M")]
    md = render_today(today=date(2026, 5, 17), recall=[], new=new)  # Sunday
    assert "## Today's hardest" not in md


def test_render_today_saturday_falls_back_to_blank_template_when_week_was_empty() -> None:
    md = render_today(
        today=date(2026, 5, 16),  # Saturday
        recall=[],
        new=[],
        hardest_marks=[],
        curriculum=[],
    )
    assert "## This week's hardest — your pick" in md
    assert md.count("- [ ] [Pattern] -> Problem Name") == 3


def test_render_today_saturday_prefills_from_past_5_weekdays_hardest_marks() -> None:
    curriculum = [
        Problem(
            text="[Heap] -> Task Scheduler",
            difficulty="M",
            link="https://neetcode.io/problems/task-scheduler",
        ),
        Problem(
            text="[Arrays & Hashing] -> Two Sum",
            difficulty="E",
            link="https://neetcode.io/problems/two-sum",
        ),
    ]
    marks = [
        HardestMark("[Heap] -> Task Scheduler", date(2026, 5, 13)),
        HardestMark("[Arrays & Hashing] -> Two Sum", date(2026, 5, 12)),
        # Outside the 5-day window — should be filtered.
        HardestMark("[Heap] -> Task Scheduler", date(2026, 5, 5)),
    ]
    md = render_today(
        today=date(2026, 5, 16),  # Saturday
        recall=[],
        new=[],
        hardest_marks=marks,
        curriculum=curriculum,
    )
    assert "## This week's hardest — your picks" in md
    assert (
        "- [ ] [Arrays & Hashing] -> [Two Sum](https://neetcode.io/problems/two-sum) (E) — flagged May 12"
        in md
    )
    assert (
        "- [ ] [Heap] -> [Task Scheduler](https://neetcode.io/problems/task-scheduler) (M) — flagged May 13"
        in md
    )
    # The 5+ day old entry is filtered.
    assert md.count("flagged") == 2


def test_render_today_saturday_shows_saturday_specific_time_blocks() -> None:
    md = render_today(today=date(2026, 5, 16), recall=[], new=[])  # Saturday
    assert "starts with this-week's-hardest sub-block" in md


def test_recompute_appends_hardest_marks_to_dedicated_ledger(tmp_path: Path) -> None:
    """End-to-end: a tick under `## Today's hardest` in today.md gets logged
    to hardest.jsonl on next recompute, and is NOT double-counted in
    completions.jsonl."""
    daily_md = tmp_path / "curriculum.md"
    daily_md.write_text(_dsa_md(two_sum_touch=None))
    today_md = tmp_path / "today.md"
    today_md.write_text(
        "## Today's hardest (2026-05-11) — pick from today's New\n\n"
        "- [x] [Arrays & Hashing] -> Two Sum (E)\n"
    )
    completions = tmp_path / "completions.jsonl"
    hardest = tmp_path / "hardest.jsonl"
    recompute(
        daily_md, today_md, completions,
        today=date(2026, 5, 12),
        hardest_ledger_path=hardest,
    )
    assert load_hardest_ledger(hardest) == [
        HardestMark("[Arrays & Hashing] -> Two Sum", date(2026, 5, 11))
    ]
    # The hardest-section tick must NOT also appear in completions.jsonl.
    assert load_ledger(completions) == []


def test_recompute_hardest_logging_is_idempotent(tmp_path: Path) -> None:
    """Re-running recompute with the same today.md doesn't duplicate marks."""
    daily_md = tmp_path / "curriculum.md"
    daily_md.write_text(_dsa_md(two_sum_touch=None))
    today_md = tmp_path / "today.md"
    today_md.write_text(
        "## Today's hardest (2026-05-11)\n\n"
        "- [x] [Arrays & Hashing] -> Two Sum (E)\n"
    )
    completions = tmp_path / "completions.jsonl"
    hardest = tmp_path / "hardest.jsonl"
    for _ in range(3):
        # render_today rewrites today.md each call, so restore the input.
        today_md.write_text(
            "## Today's hardest (2026-05-11)\n\n"
            "- [x] [Arrays & Hashing] -> Two Sum (E)\n"
        )
        recompute(
            daily_md, today_md, completions,
            today=date(2026, 5, 12),
            hardest_ledger_path=hardest,
        )
    assert len(load_hardest_ledger(hardest)) == 1


# ─── Wikilink rendering + phase-aware New header ─────────────────────────────


def test_new_section_wikilinks_the_pattern_to_its_pattern_doc() -> None:
    """Pattern label in `## New` should be an Obsidian wikilink to the
    corresponding `patterns/<slug>.md` file, so clicking opens the pattern
    note in Obsidian."""
    md = render_today(
        today=date(2026, 5, 12),
        recall=[],
        new=[
            Problem(
                text="[Arrays & Hashing] -> Contains Duplicate",
                difficulty="E",
                link="https://neetcode.io/problems/duplicate-integer",
            )
        ],
    )
    new_section = md.split("## New")[1].split("## ")[0]
    assert "[[arrays-hashing|Arrays & Hashing]]" in new_section


def test_new_section_hyperlinks_the_problem_name_to_its_external_url() -> None:
    """The problem name in `## New` should be a markdown link to the URL
    captured from curriculum.md (preserved on Problem.link)."""
    md = render_today(
        today=date(2026, 5, 12),
        recall=[],
        new=[
            Problem(
                text="[Arrays & Hashing] -> Contains Duplicate",
                difficulty="E",
                link="https://neetcode.io/problems/duplicate-integer",
            )
        ],
    )
    new_section = md.split("## New")[1].split("## ")[0]
    assert "[Contains Duplicate](https://neetcode.io/problems/duplicate-integer)" in new_section


def test_new_section_omits_external_link_when_problem_has_no_url() -> None:
    """Company-question stubs and other linkless entries should render the
    bare name (no broken markdown link), so the wikilink-to-pattern stays
    the only navigable surface."""
    md = render_today(
        today=date(2026, 5, 12),
        recall=[],
        new=[
            Problem(
                text="[Arrays & Hashing] -> Internal Stub Problem",
                difficulty="M",
                link=None,
            )
        ],
    )
    new_section = md.split("## New")[1].split("## ")[0]
    assert "-> Internal Stub Problem (M)" in new_section
    assert "Internal Stub Problem]" not in new_section  # not wrapped as a link


def test_pattern_slug_handles_1d_dp_and_2d_dp_overrides() -> None:
    """`1-D DP` and `2-D DP` map to filenames `1d-dp.md` / `2d-dp.md`, not
    the generic `1-d-dp` / `2-d-dp` slug — pinned via overrides so the
    wikilinks resolve to the actual pattern files."""
    md = render_today(
        today=date(2026, 5, 12),
        recall=[],
        new=[
            Problem(text="[1-D DP] -> Climbing Stairs", difficulty="E", link="https://neetcode.io/problems/climbing-stairs"),
            Problem(text="[2-D DP] -> Unique Paths", difficulty="M", link="https://neetcode.io/problems/unique-paths"),
        ],
    )
    assert "[[1d-dp|1-D DP]]" in md
    assert "[[2d-dp|2-D DP]]" in md


def test_new_section_header_includes_phase_budget_when_phase_is_passed() -> None:
    """When the renderer knows the current phase, the `## New` heading
    surfaces the daily new-problem budget plus phase name — so the reader
    can verify the served count matches the phase target."""
    phase = Phase(number=1, name="Linear Patterns E+M", new_per_day=5)
    md = render_today(
        today=date(2026, 5, 12),
        recall=[],
        new=[Problem("[Arrays & Hashing] -> Two Sum", difficulty="E")],
        phase=phase,
    )
    assert "## New — next from the curriculum (5/day, Phase 1 — Linear Patterns E+M)" in md


def test_new_section_header_falls_back_to_plain_when_no_phase_passed() -> None:
    """No phase context (legacy / test fixtures without a curriculum) →
    bare heading, no parenthetical."""
    md = render_today(
        today=date(2026, 5, 12),
        recall=[],
        new=[Problem("[Arrays & Hashing] -> Two Sum", difficulty="E")],
    )
    assert "## New — next from the curriculum\n" in md


def test_parse_completions_handles_wikilinked_pattern_in_ticked_lines() -> None:
    """A user ticks a `## New` line in today.md. The line now uses
    wikilink syntax for the pattern — the parser must canonicalize it
    back to the `[Pattern] -> Name` form that matches the ledger."""
    today_md = (
        "## New — next from the curriculum (5/day, Phase 1 — Linear Patterns E+M)\n\n"
        "- [x] [[arrays-hashing|Arrays & Hashing]] -> [Two Sum](https://neetcode.io/problems/two-sum) (E) ✅ 2026-05-12\n"
    )
    touches = parse_completions(today_md)
    assert len(touches) == 1
    assert touches[0].problem == "[Arrays & Hashing] -> Two Sum"
    assert touches[0].on == date(2026, 5, 12)


def test_parse_hardest_marks_handles_wikilinked_pattern_in_ticked_lines() -> None:
    """Mirror of the above for the hardest section — wikilink-formatted
    pattern label must canonicalize cleanly."""
    today_md = (
        "## Today's hardest (2026-05-12) — pick from today's New\n\n"
        "- [x] [[arrays-hashing|Arrays & Hashing]] -> [Two Sum](https://neetcode.io/problems/two-sum) (E)\n"
    )
    marks = parse_hardest_marks(today_md, fallback_date=date(2026, 5, 12))
    assert len(marks) == 1
    assert marks[0].problem == "[Arrays & Hashing] -> Two Sum"
    assert marks[0].on == date(2026, 5, 12)


# ─── Per-pattern notes stamper ────────────────────────────────────────────────

_SAMPLE_TEMPLATE = """# Sliding Window

## Trigger

Some prose.

## Canonical: [Longest Substring Without Repeating Characters](https://leetcode.com/problems/longest-substring-without-repeating-characters/)

### Mistakes

- (none yet)

## Variants

### [Best Time to Buy and Sell Stock](https://leetcode.com/problems/best-time-to-buy-and-sell-stock/)

- (none yet) — track min-so-far

### [Permutation in String](https://leetcode.com/problems/permutation-in-string/)

- (none yet)

## Why these belong together

Closing prose.
"""


def test_parse_template_problems_extracts_canonical_then_variants_in_order() -> None:
    """Stamping a notes file from a pattern template needs the list of problem
    names in document order: the Canonical heading first, then each Variant
    under `## Variants`."""
    from recall_engine import parse_template_problems

    names = parse_template_problems(_SAMPLE_TEMPLATE)

    assert names == [
        "Longest Substring Without Repeating Characters",
        "Best Time to Buy and Sell Stock",
        "Permutation in String",
    ]


def test_parse_template_problems_strips_markdown_link_syntax() -> None:
    """Variant headings wrap the name in `[Name](url)`. The stamper needs the
    bare name — `Name` — to use as the notes section heading."""
    from recall_engine import parse_template_problems

    md = "## Canonical: [Two Sum](https://example.com)\n\n## Variants\n\n### [Group Anagrams](https://example.com)\n"

    assert parse_template_problems(md) == ["Two Sum", "Group Anagrams"]


def test_parse_template_problems_returns_empty_when_template_has_no_canonical_or_variants() -> None:
    """A pattern template with no Canonical and no Variants section (e.g., a
    stub pattern doc) yields an empty list — the stamper produces nothing."""
    from recall_engine import parse_template_problems

    md = "# Some Pattern\n\n## Trigger\n\nprose only\n"

    assert parse_template_problems(md) == []


def test_stamp_notes_creates_fresh_file_with_h1_header_and_problem_stubs() -> None:
    """When no notes file exists yet, stamp_notes returns a fresh body with the
    `# <Pattern> — Notes` H1 and one `## <Problem>` stub per template problem."""
    from recall_engine import stamp_notes

    result = stamp_notes(_SAMPLE_TEMPLATE, existing=None, pattern_label="Sliding Window")

    assert result.startswith("# Sliding Window — Notes\n")
    assert "## Longest Substring Without Repeating Characters\n" in result
    assert "## Best Time to Buy and Sell Stock\n" in result
    assert "## Permutation in String\n" in result


def test_stamp_notes_preserves_every_byte_of_existing_user_content() -> None:
    """The single most important invariant: when merging into an existing
    notes file, no user-written line may be modified or removed. New stubs
    are only ever appended at the end."""
    from recall_engine import stamp_notes

    existing = (
        "# Sliding Window — Notes\n\n"
        "## Best Time to Buy and Sell Stock\n\n"
        "- set profit to 0\n"
        "- use max(profit, sell - buy)\n"
    )

    result = stamp_notes(_SAMPLE_TEMPLATE, existing=existing, pattern_label="Sliding Window")

    for line in existing.splitlines():
        assert line in result.splitlines(), f"Line lost during merge: {line!r}"


def test_stamp_notes_does_not_duplicate_problem_sections_already_present() -> None:
    """If the user already has `## Best Time to Buy and Sell Stock`, stamping
    must not append a second copy of that heading."""
    from recall_engine import stamp_notes

    existing = (
        "# Sliding Window — Notes\n\n"
        "## Best Time to Buy and Sell Stock\n\n"
        "- my notes here\n"
    )

    result = stamp_notes(_SAMPLE_TEMPLATE, existing=existing, pattern_label="Sliding Window")

    assert result.count("## Best Time to Buy and Sell Stock") == 1


def test_stamp_notes_appends_only_missing_problem_stubs() -> None:
    """Existing notes already cover one variant. The stamp adds stubs for the
    canonical + the other variant, but leaves the existing one alone."""
    from recall_engine import stamp_notes

    existing = (
        "# Sliding Window — Notes\n\n"
        "## Best Time to Buy and Sell Stock\n\n"
        "- my mistake\n"
    )

    result = stamp_notes(_SAMPLE_TEMPLATE, existing=existing, pattern_label="Sliding Window")

    assert "## Longest Substring Without Repeating Characters" in result
    assert "## Permutation in String" in result
    assert "- my mistake" in result


def test_stamp_notes_preserves_user_added_sections_unknown_to_the_template() -> None:
    """The user may have invented their own headings (e.g., `## Mistakes
    (general)`) that aren't in the template. Those must survive the merge."""
    from recall_engine import stamp_notes

    existing = (
        "# Sliding Window — Notes\n\n"
        "## Mistakes (general)\n\n"
        "- always reset state\n"
    )

    result = stamp_notes(_SAMPLE_TEMPLATE, existing=existing, pattern_label="Sliding Window")

    assert "## Mistakes (general)" in result
    assert "- always reset state" in result


def test_stamp_notes_is_idempotent_when_run_twice() -> None:
    """Running stamp twice on the same template must produce the same result —
    no extra stubs, no shuffling."""
    from recall_engine import stamp_notes

    once = stamp_notes(_SAMPLE_TEMPLATE, existing=None, pattern_label="Sliding Window")
    twice = stamp_notes(_SAMPLE_TEMPLATE, existing=once, pattern_label="Sliding Window")

    assert once == twice


def test_ensure_notes_stamped_creates_file_on_first_call(tmp_path: Path) -> None:
    """Filesystem integration: `ensure_notes_stamped` writes the stamped body
    to disk when no notes file exists yet, and returns True to signal it
    changed the world."""
    from recall_engine import ensure_notes_stamped

    template = tmp_path / "sliding-window.md"
    template.write_text(_SAMPLE_TEMPLATE)
    notes = tmp_path / "sliding-window.notes.md"

    changed = ensure_notes_stamped(template, notes, pattern_label="Sliding Window")

    assert changed is True
    assert notes.exists()
    body = notes.read_text()
    assert body.startswith("# Sliding Window — Notes")
    assert "## Best Time to Buy and Sell Stock" in body


def test_ensure_notes_stamped_is_a_noop_when_no_new_problems_to_add(tmp_path: Path) -> None:
    """If the notes file already contains every template problem, the second
    invocation must NOT rewrite the file (no spurious mtime bump)."""
    from recall_engine import ensure_notes_stamped

    template = tmp_path / "sliding-window.md"
    template.write_text(_SAMPLE_TEMPLATE)
    notes = tmp_path / "sliding-window.notes.md"

    ensure_notes_stamped(template, notes, pattern_label="Sliding Window")
    first_body = notes.read_text()
    changed_again = ensure_notes_stamped(template, notes, pattern_label="Sliding Window")
    second_body = notes.read_text()

    assert changed_again is False
    assert first_body == second_body


def test_ensure_notes_stamped_does_nothing_when_template_does_not_exist(tmp_path: Path) -> None:
    """Some patterns may not have a `patterns/<slug>.md` template yet (e.g.,
    a pattern that was added to curriculum.md before its template was
    written). The stamper must silently skip these — no crash, no notes file."""
    from recall_engine import ensure_notes_stamped

    template = tmp_path / "missing.md"
    notes = tmp_path / "missing.notes.md"

    changed = ensure_notes_stamped(template, notes, pattern_label="Missing Pattern")

    assert changed is False
    assert not notes.exists()


def test_recompute_auto_stamps_notes_for_a_pattern_on_first_tick(tmp_path: Path) -> None:
    """Integration: when a problem is ticked in today.md and `patterns_dir` is
    passed, recompute lazily creates the pattern's notes file with a fresh
    H1 + problem stubs derived from the pattern template."""
    daily = tmp_path / "curriculum.md"
    daily.write_text(THREE_DAY_CURRICULUM_MD)
    ledger = tmp_path / "completions.jsonl"
    today_md = tmp_path / "today.md"
    today_md.write_text("- [x] [A] -> P1 (Day 1) ✅ 2026-05-06\n")

    patterns_dir = tmp_path / "patterns"
    patterns_dir.mkdir()
    (patterns_dir / "a.md").write_text(
        "# A\n\n## Canonical: [P1](https://example.com)\n\n## Variants\n\n### [P2](https://example.com)\n"
    )

    recompute(daily, today_md, ledger, today=date(2026, 5, 7), patterns_dir=patterns_dir)

    notes = patterns_dir / "a.notes.md"
    assert notes.exists()
    body = notes.read_text()
    assert "# A — Notes" in body
    assert "## P1" in body
    assert "## P2" in body


def test_recompute_preserves_existing_notes_when_auto_stamping(tmp_path: Path) -> None:
    """A user's hand-written notes must survive recompute's auto-stamp pass
    untouched — only missing problem stubs are appended."""
    daily = tmp_path / "curriculum.md"
    daily.write_text(THREE_DAY_CURRICULUM_MD)
    ledger = tmp_path / "completions.jsonl"
    today_md = tmp_path / "today.md"
    today_md.write_text("- [x] [A] -> P1 (Day 1) ✅ 2026-05-06\n")

    patterns_dir = tmp_path / "patterns"
    patterns_dir.mkdir()
    (patterns_dir / "a.md").write_text(
        "# A\n\n## Canonical: [P1](https://example.com)\n\n## Variants\n\n### [P2](https://example.com)\n"
    )
    notes = patterns_dir / "a.notes.md"
    notes.write_text("# A — Notes\n\n## P1\n\n- my hard-won insight\n")

    recompute(daily, today_md, ledger, today=date(2026, 5, 7), patterns_dir=patterns_dir)

    body = notes.read_text()
    assert "- my hard-won insight" in body
    assert "## P2" in body  # the missing stub was added
    assert body.count("## P1") == 1  # no duplicate
