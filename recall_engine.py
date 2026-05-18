"""Recall engine: snapshot-mode SM-2 lite scheduler for the NeetCode 150 curriculum.

Reads `curriculum.md` (single master list — DSA grouped by phase → pattern,
plus System Design / Mocks / Behavioral sections) and the previous `today.md`
(yesterday's checked-off items), folds new completions into an append-only
JSONL ledger, then regenerates `today.md` with today's Recall and New sections.
Phase advancement is ledger-driven: the current phase is the lowest-numbered
one with eligible untouched problems.

Designed to run once per morning (LaunchAgent) or on-demand (`prep recompute`).
There is no live daemon — today's set is frozen until the next recompute call.
"""

from __future__ import annotations

import calendar
import json
import math
import re
import sys
from collections import defaultdict
from dataclasses import dataclass, replace
from datetime import date, timedelta
from pathlib import Path
from typing import Any, Iterable, Literal

import click


# ─── Constants ────────────────────────────────────────────────────────────────

INTERVALS_DAYS: list[int] = [1, 3, 7, 21]
"""SM-2 lite: interval after the Nth touch (1-indexed). 4+ touches stay at 21.

A problem touched 4 times is considered fully mastered for prep purposes —
post-acquisition retention happens via the interleaved Maintenance section
(`compute_maintenance`), which surfaces mastered items on a round-robin
cadence regardless of strict overdue status."""

DEFAULT_RECALL_LIMIT = 10
DEFAULT_NEW_LIMIT = 3

Difficulty = Literal["E", "M", "H"]
Source = Literal["nc-150", "nc-150+", "lc-only", "company question"]

_SOURCE_RANK: dict[str, int] = {
    "nc-150": 0,
    "nc-150+": 1,
    "lc-only": 1,
    "company question": 2,
}
_DIFFICULTY_RANK: dict[str, int] = {"E": 0, "M": 1, "H": 2}


# ─── Data types ──────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class Problem:
    """A curriculum entry: canonical problem text + optional metadata tags.

    `phase` is the phase number this problem was placed under in
    `curriculum.md` (set by `parse_curriculum`). None means the problem was
    parsed from a file without phase headings (legacy or test fixtures).
    `link` is the URL extracted from the markdown link in curriculum.md
    (e.g. neetcode.io / leetcode.com); preserved so renderers can re-emit it."""

    text: str
    difficulty: Difficulty | None = None
    source: Source = "nc-150"
    variant_of: str | None = None
    phase: int | None = None
    link: str | None = None

    @property
    def pattern(self) -> str:
        match = re.match(r"^\[([^\]]+)\]", self.text)
        return match.group(1) if match else ""

    @property
    def name(self) -> str:
        match = re.match(r"^\[[^\]]+\]\s*->\s*(.+)$", self.text)
        return match.group(1).strip() if match else self.text


@dataclass(frozen=True)
class Touch:
    """One successful (re-)solve event. The ledger is a list of these."""

    problem: str
    on: date


@dataclass(frozen=True)
class HardestMark:
    """User flagged this problem as the day's hardest. Logged from a tick in
    today.md's `## Today's hardest` section, persisted to hardest.jsonl, and
    surfaced on Saturday's `## This week's hardest — your picks` pre-fill."""

    problem: str
    on: date


@dataclass(frozen=True)
class RecallItem:
    """A problem ranked into the Recall section, with metadata for rendering."""

    problem: str
    touches: int
    last_touched: date
    days_overdue: int
    difficulty: Difficulty | None = None
    link: str | None = None


@dataclass(frozen=True)
class Phase:
    """A curriculum phase: an ordinal section in `curriculum.md` with a
    daily new-problem budget. Phase membership lives on the Problem itself
    (set by `parse_curriculum` from the `### Phase N — Name (X new/day)`
    headings). `new_per_day=0` puts the phase in recall-only mode."""

    number: int
    name: str
    new_per_day: int


MockStatus = Literal["pending", "scheduled", "completed"]


@dataclass(frozen=True)
class MockPrereq:
    """Per-mock readiness thresholds. `sd_chapter_ids` (when set) overrides
    `sd_chapters` — the count then derives from the list length."""

    em_problems: int = 0
    sd_chapters: int = 0
    sd_chapter_ids: tuple[str, ...] = ()

    @property
    def has_any(self) -> bool:
        return bool(self.em_problems or self.sd_chapters or self.sd_chapter_ids)


@dataclass(frozen=True)
class PrereqRow:
    """One row of a prereq summary line; `detail` is an optional inline breakdown."""

    label: str
    met: bool
    current: int
    threshold: int
    detail: str | None = None


_PLATFORM_BOOKING_URLS: dict[str, str] = {
    "Pramp": "https://www.pramp.com/dashboard/upcoming-sessions",
    "Interviewing.io": "https://interviewing.io/dashboard",
    "Hello Interview": "https://www.hellointerview.com/mock-interviews",
}


@dataclass(frozen=True)
class Mock:
    """A planned, scheduled, or completed mock interview. State machine:
    pending → scheduled (with date) → completed (with date)."""

    id: str
    status: MockStatus
    platform: str | None = None
    topic: str | None = None
    scheduled_date: date | None = None
    completed_date: date | None = None
    notes: str | None = None
    prerequisites: MockPrereq | None = None
    booking_url: str | None = None

    @property
    def effective_booking_url(self) -> str | None:
        """Per-mock override wins; otherwise the platform default. None if neither."""
        return self.booking_url or _PLATFORM_BOOKING_URLS.get(self.platform or "")

    @property
    def descriptor(self) -> str:
        """Human-readable label: `platform · topic`, or the id as fallback."""
        return " · ".join(b for b in (self.platform, self.topic) if b) or self.id


SDChapterStatus = Literal["pending", "completed"]


@dataclass(frozen=True)
class SDChapter:
    """One System Design reading unit (a book chapter). Two states: pending
    and completed. Sequential — order in the JSON file is the read order."""

    id: str
    title: str
    book: str
    status: SDChapterStatus
    completed_date: date | None = None


BehavioralStatus = Literal["pending", "completed"]


@dataclass(frozen=True)
class BehavioralTopic:
    """One behavioral interview prep unit (a story or question to prepare).
    Two states: pending and completed. The user excludes behavioral from the
    application-readiness gates — this is a self-paced checklist surfaced for
    visibility, not blocking."""

    id: str
    prompt: str
    status: BehavioralStatus
    completed_date: date | None = None
    notes: str | None = None


@dataclass(frozen=True)
class CategoryProgress:
    """Completion of one prep category (E+M, SD, mocks). Drives bars and gates."""

    name: str
    done: int
    total: int

    @property
    def fraction(self) -> float:
        return self.done / self.total if self.total > 0 else 0.0

    @property
    def complete(self) -> bool:
        return self.total > 0 and self.done == self.total


@dataclass(frozen=True)
class Readiness:
    """Raw progress counts for the three trackable categories. Rendered as
    bars in `today.md`'s `## Progress` section. No tier labels — application
    timing is the user's call (see _When to apply_ in README's Sequencing)."""

    em: CategoryProgress
    sd: CategoryProgress
    mocks: CategoryProgress


@dataclass(frozen=True)
class RecomputeResult:
    """Summary of one `recompute()` call, surfaced via the CLI."""

    new_touches_logged: int
    recall_size: int
    new_size: int
    maintenance_size: int = 0


# ─── SM-2 lite arithmetic ────────────────────────────────────────────────────


def interval_for(touches: int) -> int:
    """Days from last touch to next due. Saturates at the last entry."""
    return INTERVALS_DAYS[min(touches, len(INTERVALS_DAYS)) - 1]


def due_date(touches: int, last_touched: date) -> date:
    return last_touched + timedelta(days=interval_for(touches))


def overdue_days(touches: int, last_touched: date, today: date) -> int:
    """Positive = past due. Zero = exactly due. Negative = not yet due."""
    return (today - due_date(touches, last_touched)).days


# ─── Sprint day math ─────────────────────────────────────────────────────────


def start_date(ledger: list[Touch]) -> date | None:
    """Earliest touch in the ledger; None if empty. Anchors Day 1."""
    return min((t.on for t in ledger), default=None)


def day_n_for(today: date, start: date) -> int:
    """Day number relative to start. start_date itself is Day 1."""
    return (today - start).days + 1


# ─── Pace projection ─────────────────────────────────────────────────────────


def avg_new_per_day(ledger: list[Touch], today: date) -> float | None:
    """Distinct problems per day since start_date. None if ledger is empty."""
    start = start_date(ledger)
    if start is None:
        return None
    return len({t.problem for t in ledger}) / day_n_for(today, start)


def projected_end_date(
    ledger: list[Touch],
    curriculum: list[Problem],
    today: date,
) -> date | None:
    """Calendar date when every curriculum problem has ≥1 touch at current pace.
    None if the ledger is empty; `today` if curriculum is already complete."""
    rate = avg_new_per_day(ledger, today)
    if not rate:
        return None
    touched = {t.problem for t in ledger}
    untouched = sum(1 for p in curriculum if p.text not in touched)
    if untouched == 0:
        return today
    return today + timedelta(days=math.ceil(untouched / rate))


# ─── Touch aggregation ───────────────────────────────────────────────────────


def aggregate_touches(ledger: Iterable[Touch]) -> dict[str, tuple[int, date]]:
    """Roll the ledger up to {problem: (touch_count, latest_completion_date)}."""
    summary: dict[str, tuple[int, date]] = {}
    for t in ledger:
        count, latest = summary.get(t.problem, (0, t.on))
        summary[t.problem] = (count + 1, max(latest, t.on))
    return summary


# ─── Recall and New computation ──────────────────────────────────────────────


def compute_recall(
    ledger: list[Touch],
    today: date,
    limit: int = DEFAULT_RECALL_LIMIT,
    curriculum: list[Problem] | None = None,
) -> list[RecallItem]:
    """Items at or past their SM-2 due date, sorted most-overdue first, capped.
    If `curriculum` is given, each RecallItem is annotated with its difficulty."""
    diff_lookup: dict[str, Difficulty | None] = {
        p.text: p.difficulty for p in (curriculum or [])
    }
    link_lookup: dict[str, str | None] = {
        p.text: p.link for p in (curriculum or [])
    }
    items: list[RecallItem] = []
    for problem, (touches, last) in aggregate_touches(ledger).items():
        overdue = overdue_days(touches, last, today)
        if overdue >= 0:
            items.append(
                RecallItem(
                    problem=problem,
                    touches=touches,
                    last_touched=last,
                    days_overdue=overdue,
                    difficulty=diff_lookup.get(problem),
                    link=link_lookup.get(problem),
                )
            )
    items.sort(key=lambda r: r.days_overdue, reverse=True)
    return items[:limit]


def min_touches_in_scope(
    curriculum: list[Problem],
    ledger: list[Touch],
    difficulties: tuple[Difficulty, ...] = ("E", "M"),
) -> int:
    """Lowest touch count across curriculum problems whose difficulty is in
    `difficulties`. Returns 0 if any in-scope problem is untouched, and 0
    if the in-scope set is empty.

    Drives the post-acquisition trigger: once every E/M problem has been
    touched ≥ `_SLOT_LIMIT` times, the Maintenance section activates."""
    in_scope = [p for p in curriculum if p.difficulty in difficulties]
    if not in_scope:
        return 0
    summary = aggregate_touches(ledger)
    return min(summary.get(p.text, (0, date.min))[0] for p in in_scope)


def compute_maintenance(
    curriculum: list[Problem],
    ledger: list[Touch],
    today: date,
    limit: int = DEFAULT_RECALL_LIMIT,
    difficulties: tuple[Difficulty, ...] = ("E", "M"),
) -> list[RecallItem]:
    """Interleaved post-acquisition recall: round-robin across patterns,
    surfacing in-scope problems regardless of overdue status.

    Returns `[]` until every in-scope problem has reached `_SLOT_LIMIT`
    touches — at which point urgency-ordered `compute_recall` runs dry
    (everything sits at the 21-day cap, refreshed often enough never to
    fall overdue). Maintenance fills the gap by rotating across patterns,
    which trains the discrimination skill the interview actually rewards.

    Ordering: within each pattern bucket, least-recently-touched first.
    Across buckets, pattern-by-pattern round-robin starting from the
    largest bucket. Fully deterministic — no randomness, no seeding."""
    if min_touches_in_scope(curriculum, ledger, difficulties) < _SLOT_LIMIT:
        return []
    summary = aggregate_touches(ledger)
    buckets: dict[str, list[RecallItem]] = defaultdict(list)
    for p in curriculum:
        if p.difficulty not in difficulties:
            continue
        touches, last = summary.get(p.text, (0, date.min))
        buckets[p.pattern].append(
            RecallItem(
                problem=p.text,
                touches=touches,
                last_touched=last,
                days_overdue=overdue_days(touches, last, today),
                difficulty=p.difficulty,
                link=p.link,
            )
        )
    for pattern in buckets:
        buckets[pattern].sort(key=lambda r: r.last_touched)
    ordered_patterns = sorted(buckets, key=lambda p: (-len(buckets[p]), p))
    result: list[RecallItem] = []
    indices: dict[str, int] = {p: 0 for p in ordered_patterns}
    while len(result) < limit:
        progressed = False
        for pattern in ordered_patterns:
            if indices[pattern] < len(buckets[pattern]):
                result.append(buckets[pattern][indices[pattern]])
                indices[pattern] += 1
                progressed = True
                if len(result) >= limit:
                    return result
        if not progressed:
            return result
    return result


def compute_new(
    curriculum: list[Problem],
    ledger: list[Touch],
    limit: int = DEFAULT_NEW_LIMIT,
    phase: Phase | None = None,
) -> list[Problem]:
    """The next N never-touched problems, ordered by source provenance
    (nc-150 > nc-150+ > company question), then difficulty (E → M → H),
    then document order. When `phase` is given, the pool is filtered to
    problems assigned to that phase number in curriculum.md and the limit
    defaults to `phase.new_per_day` (so `new_per_day=0` returns nothing)."""
    touched = {t.problem for t in ledger}
    pool = [p for p in curriculum if p.text not in touched]
    if phase is not None:
        pool = [p for p in pool if p.phase == phase.number]
        limit = phase.new_per_day
    return sorted(
        pool,
        key=lambda p: (
            _SOURCE_RANK.get(p.source, 0),
            _DIFFICULTY_RANK.get(p.difficulty or "M", 1),
        ),
    )[:limit]


def current_phase(
    curriculum: list[Problem], ledger: list[Touch], phases: list[Phase]
) -> Phase | None:
    """The lowest-numbered phase that still has untouched problems assigned to
    it. Falls through to the last phase when every prior phase is drained.
    Returns None if `phases` is empty."""
    if not phases:
        return None
    touched = {t.problem for t in ledger}
    ordered = sorted(phases, key=lambda ph: ph.number)
    return next(
        (
            ph for ph in ordered
            if any(p.phase == ph.number and p.text not in touched for p in curriculum)
        ),
        ordered[-1],
    )


# ─── Parsers ─────────────────────────────────────────────────────────────────


_DSA_HEADING = re.compile(r"^##\s+(?:DSA|NeetCode\s+150\b.*)\s*$")
_PHASE_HEADING = re.compile(r"^###\s+Phase\s+(\d+)\s+—\s+(.+?)\s+\((\d+)\s+new/day\)\s*$")
_PATTERN_SUBHEADING = re.compile(r"^####\s+(.+?)\s*$")
_OTHER_H2 = re.compile(r"^##\s+(.+?)\s*$")
# Top-level DSA problem line. Accepts the new format (no parent checkbox,
# `- Two Sum (E) · 4/4 (next due 2026-06-25)`), and tolerates the legacy
# format `- [ ] Two Sum (E)` for migration. The legacy `_PROBLEM_LINE` /
# `_CHECKED_LINE` patterns are kept for parsing today.md completions.
_PROBLEM_LINE = re.compile(r"^\s*-\s+\[[ xX]\]\s+(.*)$")
_CHECKED_LINE = re.compile(r"^\s*-\s+\[[xX]\]\s+(.*)$")
_HARDEST_HEADING = re.compile(
    r"^##\s+Today's hardest(?:\s+\((\d{4}-\d{2}-\d{2})\))?(?:\s.*)?$"
)
_DSA_PROBLEM_PARENT = re.compile(r"^-\s+(?:\[[ xX]\]\s+)?(.+?)\s*$")
# Accept any `· N/M` denominator so existing curriculum.md files seeded under
# the old 5-slot mastery scheme still parse cleanly.
_DSA_COUNTER_ANNOTATION = re.compile(r"\s+·\s+\d+/\d+(?:\s+\([^)]+\))?\s*$")
_DSA_SUBBULLET = re.compile(r"^\s+-\s+\[([ xX])\](?:\s+✅\s*(\S*))?\s*$")
_DONE_DATE = re.compile(r"\s*✅\s*(\d{4}-\d{2}-\d{2}).*$")
_T_MARKER = re.compile(r"\s*`[A-Z]\d?`\s*")
_DAY_ANNOTATION = re.compile(r"\s*\(Day\s+\d+\)\s*")
_METADATA_SUFFIX = re.compile(r"\s+—\s+.*$")
_DIFFICULTY_TAG = re.compile(r"\s*\((E|M|H)\)\s*")
_SOURCE_TAG = re.compile(r"\s*\((nc-150\+|lc-only|company question)\)\s*")
_VARIANT_TAG = re.compile(r"\s*\(variant of:\s*([^)]+)\)\s*")
_MARKDOWN_LINK = re.compile(r"\[([^\]]+)\]\(([^\)]+)\)")
_WIKILINK = re.compile(r"\[\[([^|\]]+)\|([^\]]+)\]\]")
_PROBLEM_TEXT = re.compile(r"^\[[^\]]+\]\s*->\s*.+$")

# Pattern label → pattern-file slug. Generic rule: lowercase, drop `&`,
# collapse spaces to hyphens. Overrides handle the few cases where the
# pattern file diverges from the generic rule (1-D DP → 1d-dp).
_PATTERN_SLUG_OVERRIDES: dict[str, str] = {
    "1-D DP": "1d-dp",
    "2-D DP": "2d-dp",
}


def _pattern_to_slug(pattern: str) -> str:
    if pattern in _PATTERN_SLUG_OVERRIDES:
        return _PATTERN_SLUG_OVERRIDES[pattern]
    return pattern.lower().replace(" & ", "-").replace(" ", "-")


def _canonicalize(text: str) -> str:
    """Strip every non-canonical annotation: done-date stamp, em-dash metadata
    suffix, (Day N), (E)/(M)/(H), (nc-150+)/(company question), (variant of: X),
    `T2`/`M`."""
    text = _DONE_DATE.sub("", text)
    text = _METADATA_SUFFIX.sub("", text)
    text = _DAY_ANNOTATION.sub(" ", text)
    text = _DIFFICULTY_TAG.sub(" ", text)
    text = _SOURCE_TAG.sub(" ", text)
    text = _VARIANT_TAG.sub(" ", text)
    text = _T_MARKER.sub(" ", text)
    text = _WIKILINK.sub(r"[\2]", text)
    text = _MARKDOWN_LINK.sub(r"\1", text)
    return re.sub(r"\s+", " ", text).strip()


def _strip_problem_parent_annotations(body: str) -> str:
    """Strip the `· N/M (next due ...)` / `· N/M (overdue ...)` counter and
    any `✅ DATE` stamp from a parent problem line body, leaving only the
    name + difficulty/source/variant tags."""
    body = _DSA_COUNTER_ANNOTATION.sub("", body)
    body = _DONE_DATE.sub("", body)
    return body.rstrip()


def parse_curriculum(curriculum_md: str) -> list[Problem]:
    """Walk the `## DSA` section of `curriculum.md`, emit one Problem per
    top-level `- ...` line under `### Phase N — Name (X new/day)` → `#### Pattern`
    headings. Accepts the new format (`- Two Sum (E) · 4/4`) and the legacy
    checkbox format (`- [ ] Two Sum (E)`) for migration. The counter
    denominator is tolerated as any digit (`/5`, `/4`, etc.) for forward
    compatibility with curriculum files seeded under earlier mastery caps.
    Canonical text is reconstructed as `[Pattern] -> Name`."""
    problems: list[Problem] = []
    in_dsa = False
    phase_num: int | None = None
    pattern: str | None = None

    for line in curriculum_md.splitlines():
        if _DSA_HEADING.match(line):
            in_dsa = True
            phase_num = None
            pattern = None
            continue
        # Any other top-level heading exits the DSA section.
        h2 = _OTHER_H2.match(line)
        if h2 and not _DSA_HEADING.match(line):
            in_dsa = False
            phase_num = None
            pattern = None
            continue
        if not in_dsa:
            continue
        if (m := _PHASE_HEADING.match(line)):
            phase_num = int(m.group(1))
            pattern = None
            continue
        if (m := _PATTERN_SUBHEADING.match(line)):
            pattern = m.group(1).strip()
            continue
        if pattern is None:
            continue
        problem_match = _DSA_PROBLEM_PARENT.match(line)
        if not problem_match:
            continue
        raw = _strip_problem_parent_annotations(problem_match.group(1))
        diff_m = _DIFFICULTY_TAG.search(raw)
        src_m = _SOURCE_TAG.search(raw)
        var_m = _VARIANT_TAG.search(raw)
        link_m = _MARKDOWN_LINK.search(raw)
        name = _DIFFICULTY_TAG.sub(" ", raw)
        name = _SOURCE_TAG.sub(" ", name)
        name = _VARIANT_TAG.sub(" ", name)
        name = _T_MARKER.sub(" ", name)
        name = _MARKDOWN_LINK.sub(r"\1", name)
        name = re.sub(r"\s+", " ", name).strip()
        if not name:
            continue
        link = link_m.group(2) if link_m else None
        # Suppress local relative paths like ./problems/company/foo.md — those
        # are stub-file pointers, not external problem links.
        if link and link.startswith("."):
            link = None
        problems.append(
            Problem(
                text=f"[{pattern}] -> {name}",
                difficulty=diff_m.group(1) if diff_m else None,  # type: ignore[arg-type]
                source=src_m.group(1) if src_m else "nc-150",  # type: ignore[arg-type]
                variant_of=var_m.group(1).strip() if var_m else None,
                phase=phase_num,
                link=link,
            )
        )

    return problems


def parse_curriculum_dsa_state(curriculum_md: str) -> dict[str, set[date]]:
    """Walk the `## DSA` section, return `{problem_text: {touch_date, ...}}`.
    Each ticked sub-bullet `- [x] ✅ YYYY-MM-DD` adds one date to the set.
    Empty padding sub-bullets `- [ ]` are skipped silently. Sub-bullets with
    a malformed date emit a stderr warning and are skipped (non-destructive).

    Legacy migration: a `- [x] Name (E) ✅ YYYY-MM-DD` parent line is treated
    as a single touch on that date so first-recompute migration preserves
    existing ledger linkage even before sub-bullets are rendered.

    Used to sync curriculum.md ticks ↔ ledger."""
    state: dict[str, set[date]] = {}
    in_dsa = False
    pattern: str | None = None
    current_key: str | None = None

    for line in curriculum_md.splitlines():
        if _DSA_HEADING.match(line):
            in_dsa = True
            pattern = None
            current_key = None
            continue
        h2 = _OTHER_H2.match(line)
        if h2 and not _DSA_HEADING.match(line):
            in_dsa = False
            pattern = None
            current_key = None
            continue
        if not in_dsa:
            continue
        if _PHASE_HEADING.match(line):
            pattern = None
            current_key = None
            continue
        if (m := _PATTERN_SUBHEADING.match(line)):
            pattern = m.group(1).strip()
            current_key = None
            continue
        if pattern is None:
            current_key = None
            continue

        # Sub-bullet: indented `- [x]` or `- [ ]` under the current problem.
        if (sb := _DSA_SUBBULLET.match(line)):
            if current_key is None:
                continue  # orphan sub-bullet; ignore
            checked = sb.group(1).lower() == "x"
            if not checked:
                continue  # empty padding slot — silent
            date_str = sb.group(2) or ""
            try:
                state[current_key].add(date.fromisoformat(date_str))
            except ValueError:
                print(
                    f"⚠️  malformed date in sub-bullet for {current_key}: "
                    f"{line.strip()!r}",
                    file=sys.stderr,
                )
            continue

        # Top-level problem line. Accept legacy checkbox format with or
        # without a `✅ DATE` stamp.
        problem_match = _DSA_PROBLEM_PARENT.match(line)
        if not problem_match:
            current_key = None
            continue
        raw = problem_match.group(1)
        legacy_date_m = _DONE_DATE.search(raw)
        is_legacy_checked = bool(_CHECKED_LINE.match(line))
        body = _strip_problem_parent_annotations(raw)
        name = _DIFFICULTY_TAG.sub(" ", body)
        name = _SOURCE_TAG.sub(" ", name)
        name = _VARIANT_TAG.sub(" ", name)
        name = _T_MARKER.sub(" ", name)
        name = _MARKDOWN_LINK.sub(r"\1", name)
        name = re.sub(r"\s+", " ", name).strip()
        if not name:
            current_key = None
            continue
        current_key = f"[{pattern}] -> {name}"
        state.setdefault(current_key, set())
        if is_legacy_checked and legacy_date_m:
            try:
                state[current_key].add(date.fromisoformat(legacy_date_m.group(1)))
            except ValueError:
                pass
    return state


def parse_phases(curriculum_md: str) -> list[Phase]:
    """Walk the `## DSA` section, return one Phase per `### Phase N — Name (X new/day)` heading."""
    phases: list[Phase] = []
    in_dsa = False
    for line in curriculum_md.splitlines():
        if _DSA_HEADING.match(line):
            in_dsa = True
            continue
        h2 = _OTHER_H2.match(line)
        if h2 and not _DSA_HEADING.match(line):
            in_dsa = False
            continue
        if not in_dsa:
            continue
        if (m := _PHASE_HEADING.match(line)):
            phases.append(Phase(
                number=int(m.group(1)),
                name=m.group(2).strip(),
                new_per_day=int(m.group(3)),
            ))
    return phases


def parse_completions(today_md: str) -> list[Touch]:
    """Pull every checked, dated, problem-shaped line out of a today.md file.
    Skips ticks under the `## Today's hardest` section — those are flag-only
    marks, parsed separately by `parse_hardest_marks` and routed to a
    different ledger file."""
    touches: list[Touch] = []
    in_hardest = False
    for line in today_md.splitlines():
        if _HARDEST_HEADING.match(line):
            in_hardest = True
            continue
        if line.startswith("## ") and not _HARDEST_HEADING.match(line):
            in_hardest = False
        if in_hardest:
            continue
        match = _CHECKED_LINE.match(line)
        if not match:
            continue
        body = match.group(1)
        date_match = _DONE_DATE.search(body)
        if not date_match:
            continue
        canonical = _canonicalize(body)
        if _PROBLEM_TEXT.match(canonical):
            touches.append(
                Touch(problem=canonical, on=date.fromisoformat(date_match.group(1)))
            )
    return touches


def parse_hardest_marks(today_md: str, fallback_date: date) -> list[HardestMark]:
    """Pull ticked problems from the `## Today's hardest (YYYY-MM-DD)` section.
    The section heading carries the date the marks belong to (embedded by the
    renderer when today.md was generated), so flags survive any recompute
    timing — morning, evening, or multi-run days. If the heading omits the
    date (legacy / hand-edited today.md), falls back to `fallback_date`."""
    marks: list[HardestMark] = []
    in_hardest = False
    section_date = fallback_date
    for line in today_md.splitlines():
        heading_match = _HARDEST_HEADING.match(line)
        if heading_match:
            in_hardest = True
            stamped = heading_match.group(1)
            section_date = (
                date.fromisoformat(stamped) if stamped else fallback_date
            )
            continue
        if line.startswith("## ") and not heading_match:
            in_hardest = False
            continue
        if not in_hardest:
            continue
        match = _CHECKED_LINE.match(line)
        if not match:
            continue
        canonical = _canonicalize(match.group(1))
        if _PROBLEM_TEXT.match(canonical):
            marks.append(HardestMark(problem=canonical, on=section_date))
    return marks


# ─── Ledger I/O ──────────────────────────────────────────────────────────────


def _opt_date(s: str | None) -> date | None:
    return date.fromisoformat(s) if s else None


def load_ledger(path: Path) -> list[Touch]:
    """Read the JSONL ledger. Empty list if the file does not exist."""
    if not path.exists():
        return []
    touches: list[Touch] = []
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        touches.append(Touch(problem=r["problem"], on=date.fromisoformat(r["on"])))
    return touches


def load_hardest_ledger(path: Path) -> list[HardestMark]:
    """Read the hardest-marks JSONL ledger. Empty list if the file is missing."""
    if not path.exists():
        return []
    marks: list[HardestMark] = []
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        marks.append(HardestMark(problem=r["problem"], on=date.fromisoformat(r["on"])))
    return marks


def append_to_hardest_ledger(path: Path, marks: list[HardestMark]) -> int:
    """Append new HardestMark entries, deduped against existing (problem, on)
    pairs already in the file. Returns the number of new entries written."""
    existing = {(m.problem, m.on) for m in load_hardest_ledger(path)}
    fresh = [m for m in marks if (m.problem, m.on) not in existing]
    if not fresh:
        return 0
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as f:
        for m in fresh:
            f.write(json.dumps({"problem": m.problem, "on": m.on.isoformat()}) + "\n")
    return len(fresh)


def next_sd_chapter(chapters: list[SDChapter]) -> SDChapter | None:
    """First pending chapter, or None if all complete."""
    return next((ch for ch in chapters if ch.status == "pending"), None)


# ─── Curriculum-MD section parsers (SD / Mocks / Behavioral) ──────────────────


_ID_PREFIX = re.compile(r"^\[([^\]]+)\]\s+(.*)$")
_BOOK_LINK = re.compile(r"\[book\]\([^)]+\)")
_BOOKING_URL = re.compile(r"\[book\]\(([^)]+)\)")
_PREREQ = re.compile(r"prereq:\s*(.*)")
_NOTE = re.compile(r"note:\s*(.*)")
_PENDING = re.compile(r"_pending_")
_SECTION_H2 = re.compile(r"^##\s+(.+?)\s*$")


def _section_lines(md: str, name: str) -> list[str]:
    """Lines under `## <name>` up to the next `## ` heading or EOF."""
    out: list[str] = []
    in_section = False
    for line in md.splitlines():
        h = _SECTION_H2.match(line)
        if h:
            in_section = h.group(1).strip() == name
            continue
        if in_section:
            out.append(line)
    return out


def parse_sd_chapters(md: str) -> list[SDChapter]:
    """Parse the `## System Design` section of curriculum.md.
    Format: `- [ ] [id] Book · Title [✅ DATE]`."""
    out: list[SDChapter] = []
    for line in _section_lines(md, "System Design"):
        m = _PROBLEM_LINE.match(line)
        if not m:
            continue
        body = m.group(1)
        is_done = bool(_CHECKED_LINE.match(line))
        date_m = _DONE_DATE.search(body)
        body = _DONE_DATE.sub("", body).strip()
        id_m = _ID_PREFIX.match(body)
        if not id_m:
            continue
        cid, rest = id_m.group(1), id_m.group(2)
        if " · " in rest:
            book, title = rest.split(" · ", 1)
        else:
            book, title = "", rest
        out.append(SDChapter(
            id=cid,
            book=book.strip(),
            title=title.strip(),
            status="completed" if is_done else "pending",
            completed_date=date.fromisoformat(date_m.group(1)) if date_m else None,
        ))
    return out


def parse_behavioral(md: str) -> list[BehavioralTopic]:
    """Parse the `## Behavioral` section. Format: `- [ ] [id] prompt [✅ DATE] [· note: ...]`."""
    out: list[BehavioralTopic] = []
    for line in _section_lines(md, "Behavioral"):
        m = _PROBLEM_LINE.match(line)
        if not m:
            continue
        body = m.group(1)
        is_done = bool(_CHECKED_LINE.match(line))
        date_m = _DONE_DATE.search(body)
        # Extract optional note suffix.
        note: str | None = None
        if " · note: " in body:
            body, note = body.split(" · note: ", 1)
            note = note.strip()
        body = _DONE_DATE.sub("", body).strip()
        id_m = _ID_PREFIX.match(body)
        if not id_m:
            continue
        out.append(BehavioralTopic(
            id=id_m.group(1),
            prompt=id_m.group(2).strip(),
            status="completed" if is_done else "pending",
            completed_date=date.fromisoformat(date_m.group(1)) if date_m else None,
            notes=note,
        ))
    return out


def parse_mocks(md: str) -> list[Mock]:
    """Parse the `## Mocks` section.
    Format: `- [ ] [mock-id] Platform · Topic · {📅 DATE | _pending_ | ✅ DATE} · [book](url) · prereq: ... · note: ...`
    Segments after the first two are ` · `-delimited and order-independent."""
    out: list[Mock] = []
    for line in _section_lines(md, "Mocks"):
        m = _PROBLEM_LINE.match(line)
        if not m:
            continue
        body = m.group(1).strip()
        id_m = _ID_PREFIX.match(body)
        if not id_m:
            continue
        mock_id = id_m.group(1)
        rest = id_m.group(2)
        segments = [s.strip() for s in rest.split(" · ")]
        # First two segments are platform · topic (when present and not metadata).
        platform: str | None = None
        topic: str | None = None
        booking_url: str | None = None
        prereq_text: str | None = None
        note: str | None = None
        scheduled_date: date | None = None
        completed_date: date | None = None

        def is_meta(seg: str) -> bool:
            return bool(
                _BOOKING_URL.search(seg) or _DONE_DATE.search(seg) or
                _BOOKED_DATE.search(seg) or _PREREQ.match(seg) or
                _NOTE.match(seg) or _PENDING.search(seg)
            )

        head: list[str] = []
        for seg in segments:
            if not is_meta(seg) and len(head) < 2:
                head.append(seg)
                continue
            if (mh := _BOOKING_URL.search(seg)):
                booking_url = mh.group(1)
                continue
            if (dm := _DONE_DATE.search(seg)):
                completed_date = date.fromisoformat(dm.group(1))
                continue
            if (sm := _BOOKED_DATE.search(seg)):
                scheduled_date = date.fromisoformat(sm.group(1))
                continue
            if (pm := _PREREQ.match(seg)):
                prereq_text = pm.group(1).strip()
                continue
            if (nm := _NOTE.match(seg)):
                note = nm.group(1).strip()
                continue

        platform = head[0] if head else None
        topic = head[1] if len(head) > 1 else None

        is_done = bool(_CHECKED_LINE.match(line))
        if is_done:
            status: MockStatus = "completed"
        elif scheduled_date is not None:
            status = "scheduled"
        else:
            status = "pending"

        prereq = _parse_inline_prereq(prereq_text) if prereq_text else None

        out.append(Mock(
            id=mock_id,
            status=status,
            platform=platform,
            topic=topic,
            scheduled_date=scheduled_date,
            completed_date=completed_date,
            notes=note,
            prerequisites=prereq,
            booking_url=booking_url,
        ))
    return out


_INLINE_EM = re.compile(r"^(\d+)\s+E\+M$")
_INLINE_SD_COUNT = re.compile(r"^(\d+)\s+SD$")


def _parse_inline_prereq(text: str) -> MockPrereq | None:
    """Parse `15 E+M, 2 SD` or `25 E+M, axu1-4, axu1-5` into a MockPrereq."""
    em = 0
    sd_count = 0
    sd_ids: list[str] = []
    for raw in (s.strip() for s in text.split(",")):
        if not raw:
            continue
        if (m := _INLINE_EM.match(raw)):
            em = int(m.group(1))
        elif (m := _INLINE_SD_COUNT.match(raw)):
            sd_count = int(m.group(1))
        else:
            sd_ids.append(raw)
    pre = MockPrereq(em_problems=em, sd_chapters=sd_count, sd_chapter_ids=tuple(sd_ids))
    return pre if pre.has_any else None


def _render_mock_line(m: Mock) -> str:
    """One bullet for the `## Mocks` section."""
    if m.status == "completed":
        check = "x"
    else:
        check = " "
    head_bits = [f"[{m.id}]"]
    if m.platform:
        head_bits.append(m.platform)
    if m.topic:
        head_bits.append(m.topic)
    if len(head_bits) == 1:
        head = head_bits[0]
    else:
        head = f"{head_bits[0]} {' · '.join(head_bits[1:])}"
    suffix: list[str] = []
    if m.status == "scheduled" and m.scheduled_date:
        suffix.append(f"📅 {m.scheduled_date.isoformat()}")
    elif m.status == "pending":
        suffix.append("_pending_")
    elif m.status == "completed" and m.completed_date:
        suffix.append(f"✅ {m.completed_date.isoformat()}")
    if m.status != "completed" and (url := m.effective_booking_url):
        suffix.append(f"[book]({url})")
    if m.prerequisites and m.prerequisites.has_any:
        bits: list[str] = []
        if m.prerequisites.em_problems:
            bits.append(f"{m.prerequisites.em_problems} E+M")
        if m.prerequisites.sd_chapter_ids:
            bits.append(", ".join(m.prerequisites.sd_chapter_ids))
        elif m.prerequisites.sd_chapters:
            bits.append(f"{m.prerequisites.sd_chapters} SD")
        if bits:
            suffix.append(f"prereq: {', '.join(bits)}")
    if m.notes:
        suffix.append(f"note: {m.notes}")
    line = f"- [{check}] {head}"
    if suffix:
        line += " · " + " · ".join(suffix)
    return line


def write_curriculum_mocks(md: str, mocks: list[Mock]) -> str:
    """Surgically replace each mock line in curriculum.md by id (preserves any
    user-added headers/notes outside the bullet lines)."""
    by_id = {m.id: m for m in mocks}
    out: list[str] = []
    in_mocks = False
    for line in md.splitlines():
        h = _SECTION_H2.match(line)
        if h:
            in_mocks = h.group(1).strip() == "Mocks"
            out.append(line)
            continue
        if in_mocks:
            pm = _PROBLEM_LINE.match(line)
            if pm:
                body = pm.group(1).strip()
                idm = _ID_PREFIX.match(body)
                if idm and idm.group(1) in by_id:
                    out.append(_render_mock_line(by_id[idm.group(1)]))
                    continue
        out.append(line)
    return "\n".join(out)


def _atomic_write(path: Path, content: str) -> None:
    """Write via a sibling tmp file + os.replace — atomic on POSIX."""
    import os
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(content)
    os.replace(tmp, path)


def init_curriculum_file(template: Path, curriculum: Path, *, force: bool = False) -> bool:
    """Seed `curriculum` from `template` if missing (or `force=True`).

    Returns True when the file was written, False when an existing curriculum
    was preserved. Raises FileNotFoundError if `template` doesn't exist."""
    if curriculum.exists() and not force:
        return False
    _atomic_write(curriculum, template.read_text())
    return True


def ensure_runtime_dirs(*dirs: Path) -> None:
    """Create each directory if missing (idempotent)."""
    for d in dirs:
        d.mkdir(parents=True, exist_ok=True)


# ─── Per-pattern notes stamper ───────────────────────────────────────────────

_TEMPLATE_CANONICAL_RE = re.compile(r"^##\s+Canonical:\s*(.+?)\s*$")
_TEMPLATE_VARIANT_RE = re.compile(r"^###\s+(.+?)\s*$")
_VARIANTS_HEADER_RE = re.compile(r"^##\s+Variants\s*$")
_NOTES_PROBLEM_HEADING_RE = re.compile(r"^##\s+(.+?)\s*$")
_TEMPLATE_LINK_RE = re.compile(r"^\[([^\]]+)\]\([^)]+\)\s*$")
_PROBLEM_PATTERN_RE = re.compile(r"^\[([^\]]+)\]\s*->\s*")


def _strip_link(text: str) -> str:
    """`[Name](url)` → `Name`. Idempotent on text without a link."""
    m = _TEMPLATE_LINK_RE.match(text.strip())
    return m.group(1) if m else text.strip()


def parse_template_problems(template_md: str) -> list[str]:
    """Problem names from a pattern template, in document order.

    Reads the `## Canonical: [Name](url)` heading and every `### [Name](url)`
    under `## Variants`. Markdown link syntax is stripped — heading bodies
    return as plain `Name`."""
    names: list[str] = []
    in_variants = False
    for raw in template_md.splitlines():
        line = raw.rstrip()
        canonical = _TEMPLATE_CANONICAL_RE.match(line)
        if canonical:
            names.append(_strip_link(canonical.group(1)))
            in_variants = False
            continue
        if _VARIANTS_HEADER_RE.match(line):
            in_variants = True
            continue
        if in_variants and line.startswith("## "):
            in_variants = False
            continue
        if in_variants:
            variant = _TEMPLATE_VARIANT_RE.match(line)
            if variant:
                names.append(_strip_link(variant.group(1)))
    return names


def _existing_problem_headings(notes_md: str) -> set[str]:
    """Every `## ...` heading body already present in a notes file."""
    return {
        m.group(1).strip()
        for raw in notes_md.splitlines()
        for m in [_NOTES_PROBLEM_HEADING_RE.match(raw.rstrip())]
        if m
    }


def stamp_notes(template_md: str, existing: str | None, pattern_label: str) -> str:
    """Notes-file content for one pattern.

    Fresh (`existing` None/empty): `# <Pattern> — Notes` H1 + one empty
    `## <Problem>` section per template problem.

    Merge: existing content is preserved verbatim — every line, including
    any user-invented sections — and missing problem stubs are appended
    at the end."""
    problem_names = parse_template_problems(template_md)

    if not existing:
        lines = [f"# {pattern_label} — Notes", ""]
        for name in problem_names:
            lines.append(f"## {name}")
            lines.append("")
        return "\n".join(lines) + "\n"

    existing_headings = _existing_problem_headings(existing)
    missing = [n for n in problem_names if n not in existing_headings]
    if not missing:
        return existing

    body = existing if existing.endswith("\n") else existing + "\n"
    if not body.endswith("\n\n"):
        body += "\n"
    for name in missing:
        body += f"## {name}\n\n"
    return body


def _pattern_label_from_template(template_md: str) -> str | None:
    """The H1 of a pattern template — `# Sliding Window` → `Sliding Window`."""
    for raw in template_md.splitlines():
        line = raw.strip()
        if line.startswith("# ") and not line.startswith("## "):
            return line[2:].strip()
    return None


def ensure_notes_stamped(
    template_path: Path, notes_path: Path, *, pattern_label: str
) -> bool:
    """Stamp `notes_path` from `template_path` if missing; merge missing problem
    stubs if it already exists. Returns True iff content changed.

    Silently no-ops when the template doesn't exist — some curriculum patterns
    may not have a `patterns/<slug>.md` yet."""
    if not template_path.exists():
        return False
    template_md = template_path.read_text()
    existing = notes_path.read_text() if notes_path.exists() else None
    new_body = stamp_notes(template_md, existing, pattern_label)
    if new_body == existing:
        return False
    _atomic_write(notes_path, new_body)
    return True


# ─── DSA sync (curriculum.md ↔ ledger) ───────────────────────────────────────


@dataclass(frozen=True)
class DSASyncPlan:
    """Pending changes computed from comparing curriculum.md DSA state with the ledger.

    `purges` are specific `(problem, date)` Touches to remove — single-touch
    granularity, NOT a problem-wide wipe. `full_purge_problems` flags any
    problems whose entire touch history was removed in this diff, so recompute
    can keep the loud "destructive" warning for the all-sub-bullets-removed
    case."""
    additions: list[Touch]
    purges: list[Touch]
    full_purge_problems: list[str]


def diff_dsa_state(
    state: dict[str, set[date]], ledger: list[Touch], today: date
) -> DSASyncPlan:
    """Compare curriculum.md DSA tick state with the ledger at per-touch granularity.

    - Sub-bullet date in state but missing from ledger → schedule a touch on
      that date (covers fresh ticks via today.md, ticked padding slots, AND
      hand-typed backdated sub-bullets).
    - Sub-bullet date in ledger but missing from state → schedule that single
      `(problem, date)` for purge. Other touches on the same problem are
      preserved.
    - When ALL ledger entries for a problem are dropped (state[problem] = ∅
      while ledger had touches), the problem name is added to
      `full_purge_problems` so recompute keeps the destructive warning.

    Migration safeguard: if EVERY problem in `state` has an empty date set
    AND the ledger has any touches, skip purges entirely — this catches the
    common case of a fresh curriculum.md restructure where the file was
    re-rendered from a stale snapshot. To genuinely wipe DSA progress,
    leave at least one problem's sub-bullets in place (or delete the ledger
    directly)."""
    ledger_by_problem: dict[str, set[date]] = defaultdict(set)
    for t in ledger:
        ledger_by_problem[t.problem].add(t.on)

    pristine = (
        bool(state)
        and all(len(d) == 0 for d in state.values())
        and any(ledger_by_problem.values())
    )

    additions: list[Touch] = []
    purges: list[Touch] = []
    full_purge_problems: list[str] = []
    for problem, dates in state.items():
        ledger_dates = ledger_by_problem.get(problem, set())
        for d in sorted(dates - ledger_dates):
            additions.append(Touch(problem=problem, on=d))
        if pristine:
            continue
        missing = ledger_dates - dates
        if missing:
            for d in sorted(missing):
                purges.append(Touch(problem=problem, on=d))
            if not dates and ledger_dates:
                full_purge_problems.append(problem)
    return DSASyncPlan(
        additions=additions, purges=purges, full_purge_problems=full_purge_problems
    )


def purge_ledger_entries(ledger_path: Path, purge: Iterable[Touch]) -> int:
    """Remove specific `(problem, date)` Touches from the ledger. Returns
    count removed. Other touches on the same problem are preserved."""
    purge_set = {(t.problem, t.on.isoformat()) for t in purge}
    if not ledger_path.exists() or not purge_set:
        return 0
    keep: list[str] = []
    removed = 0
    for line in ledger_path.read_text().splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        if (r["problem"], r["on"]) in purge_set:
            removed += 1
            continue
        keep.append(line)
    if removed:
        _atomic_write(ledger_path, ("\n".join(keep) + "\n") if keep else "")
    return removed


_SLOT_LIMIT = len(INTERVALS_DAYS)
"""Number of mastery slots rendered per touched problem (matches SM-2 lite saturation)."""


def _annotate_due(touches: list[date], today: date) -> str:
    """Build the `(next due YYYY-MM-DD)` / `(overdue Nd)` suffix for a touched
    problem. Empty string for untouched."""
    if not touches:
        return ""
    next_due = due_date(len(touches), max(touches))
    delta = (today - next_due).days
    if delta > 0:
        return f" (overdue {delta}d)"
    return f" (next due {next_due.isoformat()})"


def write_curriculum_dsa(
    curriculum_md: str, ledger: list[Touch], today: date
) -> str:
    """Re-render every DSA problem in `curriculum.md` from the ledger.

    Each problem becomes:

        - Name (E) · {min(N, _SLOT_LIMIT)}/{_SLOT_LIMIT} (next due YYYY-MM-DD)
          - [x] ✅ DATE        (one per ledger touch, oldest → newest)
          - [ ]                (empty padding up to _SLOT_LIMIT slots, only while N < _SLOT_LIMIT)

    Untouched problems (`N == 0`) render as `- Name (E) · 0/{_SLOT_LIMIT}` with
    no sub-bullets. Past saturation (`N > _SLOT_LIMIT`), all touches render
    and there is no padding. Existing sub-bullets in the source are dropped —
    the ledger is the source of truth. Phase/pattern headings and non-DSA
    sections are preserved verbatim."""
    by_problem: dict[str, list[date]] = defaultdict(list)
    for t in ledger:
        by_problem[t.problem].append(t.on)
    for k in by_problem:
        by_problem[k].sort()

    out: list[str] = []
    in_dsa = False
    pattern: str | None = None

    for line in curriculum_md.splitlines():
        if _DSA_HEADING.match(line):
            in_dsa = True
            pattern = None
            out.append(line)
            continue
        h2 = _OTHER_H2.match(line)
        if h2 and not _DSA_HEADING.match(line):
            in_dsa = False
            pattern = None
            out.append(line)
            continue
        if not in_dsa:
            out.append(line)
            continue
        if _PHASE_HEADING.match(line):
            pattern = None
            out.append(line)
            continue
        if (m := _PATTERN_SUBHEADING.match(line)):
            pattern = m.group(1).strip()
            out.append(line)
            continue
        if pattern is None:
            out.append(line)
            continue

        # Drop existing sub-bullets; they'll be re-rendered from the ledger.
        if _DSA_SUBBULLET.match(line):
            continue

        problem_match = _DSA_PROBLEM_PARENT.match(line)
        if not problem_match:
            out.append(line)
            continue

        body = _strip_problem_parent_annotations(problem_match.group(1))
        name = _DIFFICULTY_TAG.sub(" ", body)
        name = _SOURCE_TAG.sub(" ", name)
        name = _VARIANT_TAG.sub(" ", name)
        name = _T_MARKER.sub(" ", name)
        name = _MARKDOWN_LINK.sub(r"\1", name)
        name = re.sub(r"\s+", " ", name).strip()
        if not name:
            out.append(line)
            continue

        touches = by_problem.get(f"[{pattern}] -> {name}", [])
        n = len(touches)
        out.append(f"- {body} · {min(n, _SLOT_LIMIT)}/{_SLOT_LIMIT}{_annotate_due(touches, today)}")
        for d in touches:
            out.append(f"  - [x] ✅ {d.isoformat()}")
        for _ in range(_SLOT_LIMIT - n if 0 < n < _SLOT_LIMIT else 0):
            out.append("  - [ ]")
    return "\n".join(out)


_MOCK_LINE = re.compile(r"^\s*-\s+\[([ xX])\]\s+\[([^\]]+)\]\s+(.*)$")
_BOOKED_DATE = re.compile(r"📅\s*(\d{4}-\d{2}-\d{2})")


def parse_mock_updates(
    today_md: str, known_ids: set[str]
) -> list[tuple[str, MockStatus, date]]:
    """Extract user edits to `[mock-id]` checkbox lines in today.md. ✅ → completed,
    📅 → scheduled. Lines whose id isn't in `known_ids` are skipped."""
    updates: list[tuple[str, MockStatus, date]] = []
    for line in today_md.splitlines():
        m = _MOCK_LINE.match(line)
        if not m or m.group(2) not in known_ids:
            continue
        mock_id, body = m.group(2), m.group(3)
        if hit := _DONE_DATE.search(body):
            updates.append((mock_id, "completed", date.fromisoformat(hit.group(1))))
        elif hit := _BOOKED_DATE.search(body):
            updates.append((mock_id, "scheduled", date.fromisoformat(hit.group(1))))
    return updates


def apply_mock_updates(
    mocks: list[Mock],
    updates: list[tuple[str, MockStatus, date]],
) -> tuple[list[Mock], int]:
    """Fold parsed today.md updates into the mock list. Completed status is
    sticky — a stale 📅 cannot downgrade an already-completed mock."""
    by_id = {m.id: m for m in mocks}
    changes = 0
    for mock_id, new_status, dt in updates:
        existing = by_id.get(mock_id)
        if existing is None or existing.status == "completed":
            continue
        if new_status == "completed" and existing.completed_date != dt:
            by_id[mock_id] = replace(existing, status="completed", completed_date=dt)
            changes += 1
        elif new_status == "scheduled" and (
            existing.status != "scheduled" or existing.scheduled_date != dt
        ):
            by_id[mock_id] = replace(existing, status="scheduled", scheduled_date=dt)
            changes += 1
    return list(by_id.values()), changes


def mock_prereq_status(
    mock: Mock,
    em_done: int,
    sd_done: int,
    sd_chapters: list[SDChapter] | None = None,
) -> list[PrereqRow]:
    """One `PrereqRow` per dimension a mock pins. Empty if no prereqs.
    When `sd_chapter_ids` is set, the SD row counts only those chapters and
    lists them inline — otherwise it uses the count threshold."""
    pre = mock.prerequisites
    if not pre or not pre.has_any:
        return []

    rows: list[PrereqRow] = []
    if pre.em_problems > 0:
        rows.append(PrereqRow(
            "E/M problems", em_done >= pre.em_problems, em_done, pre.em_problems
        ))

    if pre.sd_chapter_ids:
        lookup = {ch.id: ch for ch in (sd_chapters or [])}
        chapters = [(cid, lookup.get(cid)) for cid in pre.sd_chapter_ids]
        done = sum(1 for _, ch in chapters if ch and ch.status == "completed")
        detail = ", ".join(
            f"{ch.title} ✓" if ch and ch.status == "completed" else (ch.title if ch else cid)
            for cid, ch in chapters
        )
        rows.append(PrereqRow(
            "SD chapters", done >= len(chapters), done, len(chapters), detail
        ))
    elif pre.sd_chapters > 0:
        rows.append(PrereqRow(
            "SD chapters", sd_done >= pre.sd_chapters, sd_done, pre.sd_chapters
        ))

    return rows


def _check(ok: bool) -> str:
    return "✓" if ok else "❌"


def _format_prereq_line(rows: list[PrereqRow]) -> str:
    """`  - Prereqs: ✓ 5/4 E/M problems, ❌ 1/3 SD chapters: Ch 4 ✓, Ch 5, Ch 6`"""
    def fmt(r: PrereqRow) -> str:
        head = f"{_check(r.met)} {r.current}/{r.threshold} {r.label}"
        return f"{head}: {r.detail}" if r.detail else head
    return f"  - Prereqs: {', '.join(fmt(r) for r in rows)}"


def _format_urgency(delta: int) -> str:
    if delta < 0:
        return f"{abs(delta)}d ago — mark complete or reschedule"
    return {0: "today", 1: "tomorrow"}.get(delta, f"in {delta}d")




def _render_next_mock_block(
    mock: Mock,
    today: date,
    em_done: int = 0,
    sd_done: int = 0,
    sd_chapters: list[SDChapter] | None = None,
) -> list[str]:
    """Editable checkbox for the next mock, tagged `[mock-id]`.
    Pending: user appends `📅 DATE` after booking. Scheduled: user ticks box on
    completion (Tasks plugin auto-stamps ✅). Recompute folds either edit back
    into the `## Mocks` section of curriculum.md."""
    if mock.status == "scheduled" and mock.scheduled_date is not None:
        delta = (mock.scheduled_date - today).days
        line = (
            f"- [ ] [{mock.id}] {mock.descriptor} · 📅 {mock.scheduled_date.isoformat()} "
            f"({_format_date(mock.scheduled_date, weekday=True)}, {_format_urgency(delta)})"
        )
    else:
        line = f"- [ ] [{mock.id}] {mock.descriptor} — _pending_"

    lines = [line]
    prereqs = mock_prereq_status(mock, em_done, sd_done, sd_chapters)
    if prereqs:
        lines.append(_format_prereq_line(prereqs))

    if mock.status == "pending":
        if url := mock.effective_booking_url:
            lines.append(f"  - Book: [{mock.platform or 'link'}]({url})")
        lines.append(
            f"  - _Booked? `uv run prep mock schedule {mock.id} YYYY-MM-DD` "
            f"(e.g. `uv run prep mock schedule {mock.id} 2026-05-17`)._"
        )
    elif mock.status == "scheduled":
        lines.append(
            f"  - _When done, check the box above (Tasks plugin auto-stamps "
            f"`✅ DATE`) or run `uv run prep mock complete {mock.id}`._"
        )
    return lines


def append_to_ledger(path: Path, new_touches: list[Touch]) -> int:
    """Append touches to the ledger, deduping against existing entries by
    (problem, date). Returns the number of touches actually written."""
    existing = {(t.problem, t.on) for t in load_ledger(path)}
    written = 0
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as f:
        for t in new_touches:
            key = (t.problem, t.on)
            if key in existing:
                continue
            existing.add(key)
            f.write(json.dumps({"problem": t.problem, "on": t.on.isoformat()}) + "\n")
            written += 1
    return written


# ─── Renderer ────────────────────────────────────────────────────────────────


def _format_date(d: date, *, weekday: bool = False) -> str:
    """`May 11` or `Mon May 11` (weekday=True). Day is never zero-padded."""
    fmt = "%a %b " if weekday else "%b "
    return d.strftime(fmt).replace(" 0", " ") + str(d.day)


def _diff_suffix(d: Difficulty | None) -> str:
    return f" ({d})" if d else ""


def _render_problem_text(text: str, link: str | None) -> str:
    """Render canonical `[Pattern] -> Name` as a pattern wikilink plus an
    optional hyperlinked name, mirroring `_render_new_line`. Falls back to the
    raw text when the canonical format isn't recognized (legacy ledger entries
    or test fixtures without a `->` separator)."""
    match = re.match(r"^\[([^\]]+)\]\s*->\s*(.+)$", text)
    if not match:
        return text
    pattern, name = match.group(1), match.group(2)
    pattern_link = f"[[{_pattern_to_slug(pattern)}|{pattern}]]"
    name_part = f"[{name}]({link})" if link else name
    return f"{pattern_link} -> {name_part}"


def _render_recall_line(item: RecallItem) -> str:
    overdue_text = (
        "due today" if item.days_overdue == 0 else f"{item.days_overdue}d overdue"
    )
    return (
        f"- [ ] {_render_problem_text(item.problem, item.link)}"
        f"{_diff_suffix(item.difficulty)} — {overdue_text} · "
        f"{item.touches}× · last {_format_date(item.last_touched)}"
    )


def _render_maintenance_line(item: RecallItem) -> str:
    """Maintenance entries are surfaced by rotation, not urgency, so the line
    drops the overdue counter that Recall uses."""
    return (
        f"- [ ] {_render_problem_text(item.problem, item.link)}"
        f"{_diff_suffix(item.difficulty)} — "
        f"{item.touches}× · last {_format_date(item.last_touched)}"
    )


def _render_new_line(problem: Problem) -> str:
    pattern_link = f"[[{_pattern_to_slug(problem.pattern)}|{problem.pattern}]]"
    name_part = f"[{problem.name}]({problem.link})" if problem.link else problem.name
    return f"- [ ] {pattern_link} -> {name_part}{_diff_suffix(problem.difficulty)}"


def _render_hardest_pick_section(new: list[Problem], today: date) -> list[str]:
    """Mon–Fri `## Today's hardest` section: one pre-rendered checkbox per
    today's New problem. User ticks the one(s) they found hardest; next
    recompute appends to hardest.jsonl. No typing — names/links/difficulty
    are engine-rendered from the curriculum so there's no spelling risk.
    The date is embedded in the heading so parse_hardest_marks knows which
    day a tick belongs to regardless of when recompute runs."""
    if not new:
        return []
    lines = [
        f"## Today's hardest ({today.isoformat()}) — pick from today's New",
        "",
        "_Tick the problem you found hardest today. Logged to `hardest.jsonl` on next recompute; pre-fills Saturday's re-solve list._",
        "",
    ]
    for p in new:
        pattern_link = f"[[{_pattern_to_slug(p.pattern)}|{p.pattern}]]"
        name_part = f"[{p.name}]({p.link})" if p.link else p.name
        lines.append(
            f"- [ ] {pattern_link} -> {name_part}{_diff_suffix(p.difficulty)}"
        )
    return lines


def _render_saturday_hardest_picks(
    week_marks: list[HardestMark], curriculum: list[Problem], today: date
) -> list[str]:
    """Saturday `## This week's hardest` section. Pre-fills with HardestMarks
    from the past 5 weekdays (Mon–Fri preceding today) joined against the
    curriculum to recover live links/difficulty. Falls back to the empty
    3-slot template when nothing was flagged that week."""
    lookup = {p.text: p for p in curriculum}
    cutoff = today - timedelta(days=5)
    relevant = sorted(
        [m for m in week_marks if cutoff <= m.on < today and m.problem in lookup],
        key=lambda m: m.on,
    )
    if not relevant:
        return [
            "## This week's hardest — your pick",
            "",
            "_Re-solve 2-3 problems you found hardest this week. Flag them in today.md's `Today's hardest` section each weekday — they'll auto-fill here next Saturday._",
            "",
            "- [ ] [Pattern] -> Problem Name",
            "- [ ] [Pattern] -> Problem Name",
            "- [ ] [Pattern] -> Problem Name",
        ]
    lines = [
        "## This week's hardest — your picks",
        "",
        "_Re-solve each on a 30-min clock. Auto-filled from your weekday hardest flags._",
        "",
    ]
    for m in relevant:
        p = lookup[m.problem]
        link_part = f"[{p.name}]({p.link})" if p.link else p.name
        lines.append(
            f"- [ ] [{p.pattern}] -> {link_part}{_diff_suffix(p.difficulty)} "
            f"— flagged {_format_date(m.on)}"
        )
    return lines


def _render_time_blocks(mock_today: bool, *, saturday: bool = False) -> list[str]:
    """`## Time blocks` section. On Saturday, morning Recall row notes the
    this-week's-hardest sub-block. On mock days, SD shifts after the mock."""
    if saturday:
        blocks = [
            "- 9:00–13:00  Recall (starts with this-week's-hardest sub-block)",
            "- 14:00–15:30 System Design",
            "- 15:30–19:30 DSA New",
        ]
        return ["## Time blocks", "", *blocks, ""]
    blocks = (
        [
            "- 9:00–13:00  Recall",
            "- 14:00–16:00 Mock",
            "- 16:00–17:30 System Design",
            "- 17:30–19:30 DSA New",
        ]
        if mock_today
        else [
            "- 9:00–13:00  Recall",
            "- 14:00–15:30 System Design",
            "- 15:30–19:30 DSA New",
        ]
    )
    return ["## Time blocks", "", *blocks, ""]


def _format_projection_line(
    end: date, rate: float, untouched: int, today: date
) -> str:
    days_remaining = (end - today).days
    return (
        f"_Projected acquisition complete: ~{_format_date(end)}, {end.year} "
        f"({rate:.1f} new/day · {untouched} left · ~{days_remaining}d remaining)_"
    )


def render_today(
    today: date,
    recall: list[RecallItem],
    new: list[Problem],
    *,
    day_n: int | None = None,
    phase: Phase | None = None,
    total_phases: int | None = None,
    projection: date | None = None,
    projection_rate: float | None = None,
    projection_untouched: int | None = None,
    sd_next: SDChapter | None = None,
    readiness: Readiness | None = None,
    next_up_mock: Mock | None = None,
    em_done: int = 0,
    sd_done: int = 0,
    sd_chapters: list[SDChapter] | None = None,
    mock_today: bool = False,
    hardest_marks: list[HardestMark] | None = None,
    curriculum: list[Problem] | None = None,
    maintenance: list[RecallItem] | None = None,
) -> str:
    """Produce the markdown for today.md."""
    base = f"# Today — {_format_date(today, weekday=True)}, {today.year}"
    if day_n is None:
        header = f"{base} · Pre-prep"
    elif phase is None:
        header = f"{base} · Day {day_n}"
    else:
        progress = f"{phase.number}/{total_phases}" if total_phases else str(phase.number)
        header = (
            f"{base} · Phase {progress} — {phase.name} · "
            f"Day {day_n} · {phase.new_per_day} new/day"
        )

    lines = [header, ""]
    if projection and projection_rate and projection_untouched:
        lines += [
            _format_projection_line(projection, projection_rate, projection_untouched, today),
            "",
        ]
    lines += ["_Generated by `prep recompute`. Re-run anytime to refresh._", ""]
    if readiness is not None:
        lines += render_readiness_block(readiness)
    if next_up_mock is not None:
        lines += ["## Next mock", ""]
        lines += _render_next_mock_block(next_up_mock, today, em_done, sd_done, sd_chapters)
        lines.append("")

    is_saturday = today.weekday() == calendar.SATURDAY
    lines += _render_time_blocks(mock_today, saturday=is_saturday)

    lines += ["## Recall — most overdue first", ""]
    lines += (
        [_render_recall_line(i) for i in recall]
        if recall else ["_Empty — no problems are overdue yet._"]
    )
    new_header = (
        f"## New — next from the curriculum ({phase.new_per_day}/day, "
        f"Phase {phase.number} — {phase.name})"
        if phase is not None
        else "## New — next from the curriculum"
    )
    lines += ["", new_header, ""]
    lines += (
        [_render_new_line(p) for p in new]
        if new else ["_Empty — every curriculum problem has been touched at least once._"]
    )

    if sd_next is not None:
        lines += ["", "## Today's SD reading", "", f"- [ ] {sd_next.book} · {sd_next.title}"]

    if is_saturday:
        lines.append("")
        lines += _render_saturday_hardest_picks(
            hardest_marks or [], curriculum or [], today
        )
    elif new:
        # Mon–Fri: pre-rendered checkbox list of today's New problems for the
        # user to flag the day's hardest. Skip on Sundays (weekday 6); the
        # caller already gates by phase/curriculum so empty `new` skips too.
        if today.weekday() != calendar.SUNDAY:
            picks = _render_hardest_pick_section(new, today)
            if picks:
                lines += ["", *picks]

    # Maintenance sits at the bottom: ambient post-acquisition rotation,
    # below the day's prescribed work (Recall / New / SD / Hardest).
    if maintenance:
        lines += [
            "",
            "## Maintenance — interleaved across patterns",
            "",
            "_E/M at the 4-touch mastery cap, rotated by pattern. "
            "Discrimination drill, not urgency._",
            "",
        ]
        lines += [_render_maintenance_line(i) for i in maintenance]

    lines.append("")
    return "\n".join(lines)


# ─── Mock progress + upcoming ────────────────────────────────────────────────


def _progress_bar(done: int, total: int, width: int = 13) -> str:
    """Unicode block bar. Empty if total <= 0."""
    filled = round((done / total) * width) if total > 0 else 0
    return f"[{'█' * filled}{'░' * (width - filled)}]"


def _section_header(name: str, done: int, total: int, extra: str = "") -> list[str]:
    """`## Name (X/Y complete[ · extra])` + blank + progress bar + blank."""
    summary = f"{done}/{total} complete" + (f" · {extra}" if extra else "")
    return [
        f"## {name} ({summary})",
        "",
        f"{_progress_bar(done, total)} {done}/{total}",
        "",
    ]


def next_mock(mocks: list[Mock]) -> Mock | None:
    """First non-completed mock, or None if all completed."""
    return next((m for m in mocks if m.status != "completed"), None)


def _format_mock_line(mock: Mock) -> str:
    """One bullet describing a single mock — its status, date, platform, topic."""
    if mock.status == "completed" and mock.completed_date:
        return f"- [x] {mock.descriptor}{_stamp(mock.completed_date)}"
    if mock.status == "scheduled" and mock.scheduled_date:
        return f"- [ ] {mock.descriptor} · 📅 {mock.scheduled_date.isoformat()}"
    return f"- [ ] {mock.descriptor} · _pending_"


# ─── Progress bars ───────────────────────────────────────────────────────────


def compute_readiness(
    curriculum: list[Problem],
    ledger: list[Touch],
    sd_chapters: list[SDChapter],
    mocks: list[Mock],
) -> Readiness:
    """Roll up the three ledgers into category progress bars (E+M / SD / Mocks)."""
    touched = {t.problem for t in ledger}
    em_curr = [p for p in curriculum if p.difficulty in ("E", "M")]
    em = CategoryProgress("E+M problems", sum(1 for p in em_curr if p.text in touched), len(em_curr))
    sd = CategoryProgress("System Design", sum(1 for c in sd_chapters if c.status == "completed"), len(sd_chapters))
    mp = CategoryProgress("Mocks", sum(1 for m in mocks if m.status == "completed"), len(mocks))
    return Readiness(em=em, sd=sd, mocks=mp)


def _render_category_line(cat: CategoryProgress) -> str:
    pct = round(cat.fraction * 100)
    return f"- {cat.name:<14} {_progress_bar(cat.done, cat.total)} {pct}% ({cat.done}/{cat.total})"


def render_readiness_block(readiness: Readiness) -> list[str]:
    """`## Progress` block: three category bars (E+M / SD / Mocks). No tier
    gates — application timing is the user's judgment call, not the engine's."""
    lines = ["## Progress", ""]
    lines += [_render_category_line(c) for c in (readiness.em, readiness.sd, readiness.mocks)]
    lines.append("")
    return lines


# ─── Orchestration ───────────────────────────────────────────────────────────


def recompute(
    curriculum_md_path: Path,
    today_md_path: Path,
    ledger_path: Path,
    today: date,
    recall_limit: int = DEFAULT_RECALL_LIMIT,
    dry_run: bool = False,
    hardest_ledger_path: Path | None = None,
    patterns_dir: Path | None = None,
) -> RecomputeResult:
    """One full cycle: log new completions, fold mock edits, regenerate today.md.

    Reads everything (DSA, SD, Mocks, Behavioral) from curriculum.md. When
    today.md ticks update mock state, the change is written back to
    curriculum.md atomically. DSA boxes ticked directly in curriculum.md log
    a touch dated today; unticked boxes purge ledger entries (warned).
    `dry_run=True` previews destructive purges instead of applying them."""
    curriculum_md = curriculum_md_path.read_text()
    curriculum = parse_curriculum(curriculum_md)
    phases = parse_phases(curriculum_md)
    sd_chapters = parse_sd_chapters(curriculum_md)
    mocks = parse_mocks(curriculum_md)
    behavioral = parse_behavioral(curriculum_md)

    today_md = today_md_path.read_text() if today_md_path.exists() else ""

    # Diff curriculum.md DSA ticks against the prior ledger BEFORE applying
    # today.md touches — otherwise a today.md tick would race ahead of the
    # curriculum.md box and look like a user uncheck.
    prev_ledger = load_ledger(ledger_path)
    dsa_state = parse_curriculum_dsa_state(curriculum_md)
    plan = diff_dsa_state(dsa_state, prev_ledger, today)
    if plan.full_purge_problems:
        # The destructive path: a problem's entire touch history was removed
        # by deleting all its sub-bullets. Loud warning, even on dry-run.
        if dry_run:
            click.echo(
                f"[dry-run] Would fully purge {len(plan.full_purge_problems)} "
                f"problem(s):",
                err=True,
            )
        else:
            click.echo(
                f"⚠️  Full purge: {len(plan.full_purge_problems)} problem(s) "
                f"had every ledger entry removed. To preview before applying, "
                f"use `prep recompute --dry-run`.",
                err=True,
            )
        for p in plan.full_purge_problems:
            click.echo(f"  - {p}", err=True)

    if plan.purges and not dry_run:
        purge_ledger_entries(ledger_path, plan.purges)
    elif plan.purges and dry_run:
        click.echo(
            f"[dry-run] Would remove {len(plan.purges)} individual touch(es) "
            f"from the ledger.",
            err=True,
        )

    today_completions = parse_completions(today_md)
    logged = append_to_ledger(ledger_path, today_completions)
    if plan.additions and not dry_run:
        logged += append_to_ledger(ledger_path, plan.additions)

    # Auto-stamp per-pattern notes for any pattern whose problems were touched
    # this cycle. Idempotent: no-op once the notes file already covers every
    # template problem. Preserves all existing user content.
    if patterns_dir is not None and not dry_run:
        touched_problems = (
            {t.problem for t in today_completions}
            | {t.problem for t in plan.additions}
        )
        touched_patterns: set[str] = set()
        for pstr in touched_problems:
            m = _PROBLEM_PATTERN_RE.match(pstr)
            if m:
                touched_patterns.add(m.group(1))
        for label in touched_patterns:
            slug = _pattern_to_slug(label)
            ensure_notes_stamped(
                patterns_dir / f"{slug}.md",
                patterns_dir / f"{slug}.notes.md",
                pattern_label=label,
            )

    # Hardest marks: parse ticks under `## Today's hardest (DATE)`. The date
    # in the heading (rendered when today.md was generated) tells us which
    # day each mark belongs to — robust against multi-recompute days or
    # evening reruns. Fallback to `today` only if the heading is dateless
    # (legacy / hand-edited).
    if hardest_ledger_path is not None and not dry_run:
        marks = parse_hardest_marks(today_md, fallback_date=today)
        if marks:
            append_to_hardest_ledger(hardest_ledger_path, marks)

    ledger = load_ledger(ledger_path)
    recall = compute_recall(ledger, today=today, limit=recall_limit, curriculum=curriculum)
    phase = current_phase(curriculum, ledger, phases) if phases else None
    new = (
        compute_new(curriculum, ledger, phase=phase) if phase
        else compute_new(curriculum, ledger)
    )
    maintenance = compute_maintenance(curriculum, ledger, today=today, limit=recall_limit)

    start = start_date(ledger)
    day_n = day_n_for(today, start) if start else None

    rate = avg_new_per_day(ledger, today)
    end = projected_end_date(ledger, curriculum, today)
    touched = {t.problem for t in ledger}
    untouched = sum(1 for p in curriculum if p.text not in touched)

    if mocks:
        updates = parse_mock_updates(today_md, {m.id for m in mocks})
        if updates:
            mocks, _ = apply_mock_updates(mocks, updates)

    # One atomic write covers mock state + DSA tick state. Preserve the
    # original trailing newline (or lack thereof) to avoid spurious diffs.
    new_md = write_curriculum_mocks(curriculum_md, mocks) if mocks else curriculum_md
    new_md = write_curriculum_dsa(new_md, ledger, today)
    if curriculum_md.endswith("\n") and not new_md.endswith("\n"):
        new_md += "\n"
    if not dry_run and new_md != curriculum_md:
        _atomic_write(curriculum_md_path, new_md)

    readiness = compute_readiness(curriculum, ledger, sd_chapters, mocks)
    em_done, sd_done = readiness.em.done, readiness.sd.done

    mock_today = any(m.scheduled_date == today for m in mocks)
    hardest_marks = (
        load_hardest_ledger(hardest_ledger_path)
        if hardest_ledger_path is not None else []
    )
    today_md_path.write_text(render_today(
        today=today,
        recall=recall,
        new=new,
        day_n=day_n,
        phase=phase,
        total_phases=len(phases) if phases else None,
        projection=end,
        projection_rate=rate,
        projection_untouched=untouched,
        sd_next=next_sd_chapter(sd_chapters),
        readiness=readiness,
        next_up_mock=next_mock(mocks),
        em_done=em_done,
        sd_done=sd_done,
        sd_chapters=sd_chapters,
        mock_today=mock_today,
        hardest_marks=hardest_marks,
        curriculum=curriculum,
        maintenance=maintenance,
    ))

    return RecomputeResult(
        new_touches_logged=logged,
        recall_size=len(recall),
        new_size=len(new),
        maintenance_size=len(maintenance),
    )


# ─── CLI ─────────────────────────────────────────────────────────────────────


@click.group()
def cli() -> None:
    """Snapshot-mode recall engine for the NeetCode 150 curriculum."""


@cli.command(name="recompute")
@click.option(
    "--curriculum",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    default=Path("curriculum.md"),
    show_default=True,
    help="Master curriculum markdown (DSA by phase + SD/Mocks/Behavioral).",
)
@click.option(
    "--today-md",
    type=click.Path(dir_okay=False, path_type=Path),
    default=Path("today.md"),
    show_default=True,
    help="Generated daily list. Overwritten on every recompute.",
)
@click.option(
    "--ledger",
    type=click.Path(dir_okay=False, path_type=Path),
    default=Path("prep-data/completions.jsonl"),
    show_default=True,
    help="Append-only completion ledger.",
)
@click.option(
    "--hardest-ledger",
    type=click.Path(dir_okay=False, path_type=Path),
    default=Path("prep-data/hardest.jsonl"),
    show_default=True,
    help="Append-only hardest-of-day flag ledger; pre-fills Saturday's re-solve list.",
)
@click.option(
    "--dry-run",
    is_flag=True,
    help="Preview destructive ledger purges (from DSA unchecks in curriculum.md) without applying.",
)
@click.option(
    "--patterns-dir",
    type=click.Path(file_okay=False, path_type=Path),
    default=Path("patterns"),
    show_default=True,
    help="Directory of `<slug>.md` pattern templates. Touched patterns get their `.notes.md` stamped (preserving existing notes).",
)
def recompute_cmd(
    curriculum: Path,
    today_md: Path,
    ledger: Path,
    hardest_ledger: Path,
    dry_run: bool,
    patterns_dir: Path,
) -> None:
    """Fold yesterday's checks into the ledger, then regenerate today.md."""
    result = recompute(
        curriculum,
        today_md,
        ledger,
        today=date.today(),
        dry_run=dry_run,
        hardest_ledger_path=hardest_ledger,
        patterns_dir=patterns_dir if patterns_dir.exists() else None,
    )
    maint_suffix = (
        f" · maintenance: {result.maintenance_size}" if result.maintenance_size else ""
    )
    click.echo(
        f"logged {result.new_touches_logged} new touch(es) · "
        f"recall: {result.recall_size} · new: {result.new_size}{maint_suffix}"
    )


def schedule_mock(curriculum_md: str, mock_id: str, dt: date) -> str:
    """Return curriculum.md with `mock_id` flipped to scheduled on `dt`.
    Raises ValueError for unknown ids or mocks that are already completed."""
    mocks = parse_mocks(curriculum_md)
    by_id = {m.id: m for m in mocks}
    if mock_id not in by_id:
        raise ValueError(
            f"unknown mock id: {mock_id!r}. known: {', '.join(sorted(by_id)) or '(none)'}"
        )
    if by_id[mock_id].status == "completed":
        raise ValueError(f"mock {mock_id!r} is already completed; cannot reschedule.")
    updated = [
        replace(m, status="scheduled", scheduled_date=dt) if m.id == mock_id else m
        for m in mocks
    ]
    return write_curriculum_mocks(curriculum_md, updated)


def complete_mock(curriculum_md: str, mock_id: str, dt: date) -> str:
    """Return curriculum.md with `mock_id` marked completed on `dt`.
    Raises ValueError for unknown ids. Idempotent for already-completed mocks
    with the same completion date."""
    mocks = parse_mocks(curriculum_md)
    by_id = {m.id: m for m in mocks}
    if mock_id not in by_id:
        raise ValueError(
            f"unknown mock id: {mock_id!r}. known: {', '.join(sorted(by_id)) or '(none)'}"
        )
    updated = [
        replace(m, status="completed", completed_date=dt) if m.id == mock_id else m
        for m in mocks
    ]
    return write_curriculum_mocks(curriculum_md, updated)


def _next_weekday_of(today: date, kind: str) -> date:
    """Return the next date matching `kind` (`weekday` = Mon–Fri, `sat`, `sun`).
    Returns `today` itself if it already matches."""
    if kind == "sat":
        target = calendar.SATURDAY
    elif kind == "sun":
        target = calendar.SUNDAY
    elif kind == "weekday":
        d = today
        while d.weekday() >= calendar.SATURDAY:
            d += timedelta(days=1)
        return d
    else:
        raise click.BadParameter(f"unknown day kind: {kind!r}")
    delta = (target - today.weekday()) % 7
    return today + timedelta(days=delta)


def _render_preview(
    curriculum_md_path: Path,
    ledger_path: Path,
    hardest_ledger_path: Path | None,
    today: date,
) -> str:
    """Read-only render of today.md for a synthetic date. No writes, no ledger
    mutation, no curriculum tick folding — pure projection of current state."""
    curriculum_md = curriculum_md_path.read_text()
    curriculum = parse_curriculum(curriculum_md)
    phases = parse_phases(curriculum_md)
    sd_chapters = parse_sd_chapters(curriculum_md)
    mocks = parse_mocks(curriculum_md)

    ledger = load_ledger(ledger_path) if ledger_path.exists() else []
    recall = compute_recall(ledger, today=today, limit=DEFAULT_RECALL_LIMIT, curriculum=curriculum)
    phase = current_phase(curriculum, ledger, phases) if phases else None
    new = (
        compute_new(curriculum, ledger, phase=phase) if phase
        else compute_new(curriculum, ledger)
    )

    start = start_date(ledger)
    day_n = day_n_for(today, start) if start else None
    rate = avg_new_per_day(ledger, today)
    end = projected_end_date(ledger, curriculum, today)
    touched = {t.problem for t in ledger}
    untouched = sum(1 for p in curriculum if p.text not in touched)

    readiness = compute_readiness(curriculum, ledger, sd_chapters, mocks)
    mock_today = any(m.scheduled_date == today for m in mocks)
    hardest_marks = (
        load_hardest_ledger(hardest_ledger_path)
        if hardest_ledger_path is not None and hardest_ledger_path.exists() else []
    )
    return render_today(
        today=today,
        recall=recall,
        new=new,
        day_n=day_n,
        phase=phase,
        total_phases=len(phases) if phases else None,
        projection=end,
        projection_rate=rate,
        projection_untouched=untouched,
        sd_next=next_sd_chapter(sd_chapters),
        readiness=readiness,
        next_up_mock=next_mock(mocks),
        em_done=readiness.em.done,
        sd_done=readiness.sd.done,
        sd_chapters=sd_chapters,
        mock_today=mock_today,
        hardest_marks=hardest_marks,
        curriculum=curriculum,
    )


@cli.command(name="preview")
@click.option(
    "--curriculum",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    default=Path("curriculum.md"),
    show_default=True,
    help="Master curriculum markdown (DSA by phase + SD/Mocks/Behavioral).",
)
@click.option(
    "--ledger",
    type=click.Path(dir_okay=False, path_type=Path),
    default=Path("prep-data/completions.jsonl"),
    show_default=True,
    help="Append-only completion ledger.",
)
@click.option(
    "--hardest-ledger",
    type=click.Path(dir_okay=False, path_type=Path),
    default=Path("prep-data/hardest.jsonl"),
    show_default=True,
    help="Hardest-of-day flag ledger (pre-fills Saturday's re-solve list).",
)
@click.option(
    "--for",
    "for_day",
    type=click.Choice(["weekday", "sat", "sun"]),
    help="Preview the next occurrence of this day. Mutually exclusive with --date.",
)
@click.option(
    "--date",
    "for_date",
    type=click.DateTime(formats=["%Y-%m-%d"]),
    help="Preview for a specific date (YYYY-MM-DD). Mutually exclusive with --for.",
)
def preview_cmd(
    curriculum: Path,
    ledger: Path,
    hardest_ledger: Path,
    for_day: str | None,
    for_date: Any | None,
) -> None:
    """Print what today.md would look like for a given day. Read-only — does not
    touch the ledger or overwrite today.md."""
    if for_day and for_date:
        raise click.UsageError("--for and --date are mutually exclusive.")
    if for_date is not None:
        target = for_date.date()
    elif for_day:
        target = _next_weekday_of(date.today(), for_day)
    else:
        target = date.today()
    click.echo(_render_preview(curriculum, ledger, hardest_ledger, target))


@cli.command(name="reset")
@click.option("--yes", is_flag=True, help="Skip the confirmation prompt.")
@click.option(
    "--ledger-dir",
    type=click.Path(file_okay=False, path_type=Path),
    default=Path("prep-data"),
    show_default=True,
    help="Directory containing the ledger.",
)
def reset_cmd(yes: bool, ledger_dir: Path) -> None:
    """Delete the DSA completion ledger. Next recompute treats you as pre-prep."""
    path = ledger_dir / "completions.jsonl"
    existence = "exists" if path.exists() else "does not exist (no-op)"
    click.echo(f"Will delete: {path} ({existence})")
    if not yes and not click.confirm("Proceed?", default=False):
        click.echo("Aborted.")
        return
    if path.exists():
        path.unlink()
    click.echo("Reset complete. Run `prep recompute` to regenerate today.md.")


@cli.command(name="init")
@click.option(
    "--template",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    default=Path("curriculum.template.md"),
    show_default=True,
    help="Pristine curriculum template (committed to the repo).",
)
@click.option(
    "--curriculum",
    type=click.Path(dir_okay=False, path_type=Path),
    default=Path("curriculum.md"),
    show_default=True,
    help="Your local working curriculum (gitignored).",
)
@click.option(
    "--force",
    is_flag=True,
    help="Overwrite curriculum.md even if it already exists.",
)
def init_cmd(template: Path, curriculum: Path, force: bool) -> None:
    """Seed your local curriculum.md from the pristine template.

    Idempotent: leaves an existing curriculum.md alone unless --force.
    Also creates prep-data/ and problems/ for the ledgers and your solutions.

    Per-pattern annotations belong in `patterns/<slug>.notes.md` (gitignored,
    lazy-created). The `patterns/<slug>.md` files stay pristine templates.
    """
    ensure_runtime_dirs(Path("prep-data"), Path("problems"))
    wrote = init_curriculum_file(template, curriculum, force=force)
    if wrote:
        click.echo(f"Seeded {curriculum} from {template}.")
        click.echo("Next: `uv run prep recompute` to generate today.md.")
    else:
        click.echo(
            f"{curriculum} already exists — left untouched. "
            f"Pass --force to overwrite from the template."
        )


@cli.group(name="notes")
def notes_cmd() -> None:
    """Manage per-pattern personal notes files (gitignored)."""


@notes_cmd.command(name="init")
@click.argument("pattern", required=False)
@click.option(
    "--patterns-dir",
    type=click.Path(exists=True, file_okay=False, path_type=Path),
    default=Path("patterns"),
    show_default=True,
    help="Directory containing `<slug>.md` pattern templates.",
)
def notes_init_cmd(pattern: str | None, patterns_dir: Path) -> None:
    """Stamp a `<slug>.notes.md` file with `## <Problem>` headings from its template.

    Pass a slug (e.g., `sliding-window`) to stamp one pattern, or omit the
    argument to stamp every template under `patterns/`. Existing notes content
    is preserved verbatim — only missing problem stubs are appended.
    """
    if pattern:
        slugs = [pattern]
    else:
        slugs = sorted(
            p.stem for p in patterns_dir.glob("*.md")
            if not p.name.endswith(".notes.md")
        )

    stamped = 0
    for slug in slugs:
        tpath = patterns_dir / f"{slug}.md"
        if not tpath.exists():
            click.echo(f"  skip {slug}: no template at {tpath}", err=True)
            continue
        label = _pattern_label_from_template(tpath.read_text())
        if not label:
            click.echo(f"  skip {slug}: template missing `# Pattern` H1", err=True)
            continue
        npath = patterns_dir / f"{slug}.notes.md"
        if ensure_notes_stamped(tpath, npath, pattern_label=label):
            stamped += 1
            click.echo(f"  stamped {npath}")
    click.echo(f"{stamped} notes file(s) stamped or updated.")


@cli.group(name="mock")
def mock_cmd() -> None:
    """Schedule, complete, and list mock interviews."""


@mock_cmd.command(name="schedule")
@click.argument("mock_id")
@click.argument("scheduled", type=click.DateTime(formats=["%Y-%m-%d"]))
@click.option(
    "--curriculum",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    default=Path("curriculum.md"),
    show_default=True,
    help="Master curriculum markdown (DSA by phase + SD/Mocks/Behavioral).",
)
def mock_schedule_cmd(mock_id: str, scheduled: Any, curriculum: Path) -> None:
    """Schedule MOCK_ID for SCHEDULED date (YYYY-MM-DD).

    Example: prep mock schedule mock-1 2026-05-17
    """
    target = scheduled.date()
    md = curriculum.read_text()
    try:
        new_md = schedule_mock(md, mock_id, target)
    except ValueError as exc:
        raise click.UsageError(str(exc))
    if new_md != md:
        _atomic_write(curriculum, new_md)
    click.echo(f"Scheduled {mock_id} for {target.isoformat()}.")


@mock_cmd.command(name="complete")
@click.argument("mock_id")
@click.option(
    "--date",
    "completion_date",
    type=click.DateTime(formats=["%Y-%m-%d"]),
    default=None,
    help="Completion date (YYYY-MM-DD). Defaults to today.",
)
@click.option(
    "--curriculum",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    default=Path("curriculum.md"),
    show_default=True,
    help="Master curriculum markdown.",
)
def mock_complete_cmd(mock_id: str, completion_date: Any | None, curriculum: Path) -> None:
    """Mark MOCK_ID as completed (defaults to today).

    Example: prep mock complete mock-1
    """
    target = completion_date.date() if completion_date is not None else date.today()
    md = curriculum.read_text()
    try:
        new_md = complete_mock(md, mock_id, target)
    except ValueError as exc:
        raise click.UsageError(str(exc))
    if new_md != md:
        _atomic_write(curriculum, new_md)
    click.echo(f"Marked {mock_id} complete as of {target.isoformat()}.")


@mock_cmd.command(name="list")
@click.option(
    "--curriculum",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    default=Path("curriculum.md"),
    show_default=True,
    help="Master curriculum markdown.",
)
def mock_list_cmd(curriculum: Path) -> None:
    """Print every mock and its current state."""
    mocks = parse_mocks(curriculum.read_text())
    if not mocks:
        click.echo("No mocks defined.")
        return
    for m in mocks:
        if m.status == "completed":
            tail = f"✅ {m.completed_date.isoformat() if m.completed_date else ''}"
        elif m.status == "scheduled":
            tail = f"📅 {m.scheduled_date.isoformat() if m.scheduled_date else ''}"
        else:
            tail = "_pending_"
        click.echo(f"{m.id:14} {tail:18} {m.descriptor}")


if __name__ == "__main__":
    cli()
