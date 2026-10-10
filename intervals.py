from datetime import datetime, timedelta, time
from typing import Optional, List, Tuple
from dataclasses import dataclass
import pytz


@dataclass(frozen=True)
class TimeInterval:
    """
    Immutable representation of a time interval [start, end).
    Both start and end should be timezone-aware (preferably UTC).
    """
    start: datetime
    end: datetime

    def __post_init__(self):
        s = self.start
        e = self.end
        if s.tzinfo is None:
            s = pytz.UTC.localize(s)
            object.__setattr__(self, "start", s)
        if e.tzinfo is None:
            e = pytz.UTC.localize(e)
            object.__setattr__(self, "end", e)

        if s >= e:
            raise ValueError(f"Interval start ({s}) must be before end ({e})")

    @property
    def duration_minutes(self) -> float:
        return (self.end - self.start).total_seconds() / 60.0

    def overlaps(self, other: "TimeInterval") -> bool:
        """
        Returns True if two intervals overlap: max(s1, s2) < min(e1, e2)
        Edge-to-edge (e.g. 10:00-11:00 and 11:00-12:00) is NOT considered an overlap.
        """
        return max(self.start, other.start) < min(self.end, other.end)

    def intersection(self, other: "TimeInterval") -> Optional["TimeInterval"]:
        """Returns the overlapping sub-interval, or None if disjoint."""
        latest_start = max(self.start, other.start)
        earliest_end = min(self.end, other.end)
        if latest_start < earliest_end:
            return TimeInterval(latest_start, earliest_end)
        return None

    def gap_minutes(self, other: "TimeInterval") -> float:
        """
        Returns the gap in minutes between this interval and another.
        Returns 0.0 if they touch or overlap.
        """
        if self.overlaps(other):
            return 0.0
        if self.end <= other.start:
            return (other.start - self.end).total_seconds() / 60.0
        else:
            return (self.start - other.end).total_seconds() / 60.0


def check_interval_overlap(int_a: TimeInterval, int_b: TimeInterval) -> bool:
    """Convenience functional check for overlap."""
    return int_a.overlaps(int_b)


def calculate_buffer_minutes(candidate: TimeInterval, busy_intervals: List[TimeInterval]) -> Tuple[float, float, int]:
    """
    Calculates:
    1. buffer_before: minutes since the immediately preceding meeting (capped at 180 mins)
    2. buffer_after: minutes until the immediately following meeting (capped at 180 mins)
    3. consecutive_meetings: how many back-to-back meetings are adjacent without at least 15 min break
    """
    if not busy_intervals:
        return (180.0, 180.0, 0)

    # Sort busy intervals chronologically
    sorted_busy = sorted(busy_intervals, key=lambda x: x.start)
    
    # Preceding meetings (end <= candidate.start)
    preceding = [b for b in sorted_busy if b.end <= candidate.start]
    # Subsequent meetings (start >= candidate.end)
    subsequent = [b for b in sorted_busy if b.start >= candidate.end]

    buffer_before = 180.0
    consecutive_before = 0
    if preceding:
        last_prec = preceding[-1]
        buffer_before = (candidate.start - last_prec.end).total_seconds() / 60.0
        
        # Check back-to-back chain before
        curr_gap = buffer_before
        idx = len(preceding) - 1
        while idx >= 0 and curr_gap < 15.0:
            consecutive_before += 1
            if idx > 0:
                curr_gap = (preceding[idx].start - preceding[idx - 1].end).total_seconds() / 60.0
            idx -= 1

    buffer_after = 180.0
    consecutive_after = 0
    if subsequent:
        first_sub = subsequent[0]
        buffer_after = (first_sub.start - candidate.end).total_seconds() / 60.0
        
        # Check back-to-back chain after
        curr_gap = buffer_after
        idx = 0
        while idx < len(subsequent) and curr_gap < 15.0:
            consecutive_after += 1
            if idx + 1 < len(subsequent):
                curr_gap = (subsequent[idx + 1].start - subsequent[idx].end).total_seconds() / 60.0
            idx += 1

    total_consecutive = consecutive_before + consecutive_after
    return (min(buffer_before, 180.0), min(buffer_after, 180.0), total_consecutive)


def is_within_working_hours(
    candidate: TimeInterval,
    user_tz_str: str,
    work_start_hour: int,
    work_end_hour: int
) -> Tuple[bool, str]:
    """
    Converts candidate UTC interval to user's local timezone
    and verifies if it falls completely within their working hours.
    """
    try:
        user_tz = pytz.timezone(user_tz_str)
    except Exception:
        user_tz = pytz.UTC

    local_start = candidate.start.astimezone(user_tz)
    local_end = candidate.end.astimezone(user_tz)

    # Work boundaries on that local date
    day_work_start = local_start.replace(hour=work_start_hour, minute=0, second=0, microsecond=0)
    day_work_end = local_start.replace(hour=work_end_hour, minute=0, second=0, microsecond=0)

    if local_start < day_work_start or local_end > day_work_end:
        return (
            False,
            f"Outside working hours ({work_start_hour}:00-{work_end_hour}:00 {user_tz_str}). "
            f"Local meeting time: {local_start.strftime('%H:%M')}-{local_end.strftime('%H:%M')}"
        )
    return (True, "")


def overlaps_lunch(
    candidate: TimeInterval,
    user_tz_str: str,
    lunch_start_hour: int,
    lunch_end_hour: int
) -> bool:
    """
    Checks if the candidate slot overlaps with the user's local lunch hour.
    """
    try:
        user_tz = pytz.timezone(user_tz_str)
    except Exception:
        user_tz = pytz.UTC

    local_start = candidate.start.astimezone(user_tz)
    local_end = candidate.end.astimezone(user_tz)

    local_lunch_start = local_start.replace(hour=lunch_start_hour, minute=0, second=0, microsecond=0)
    local_lunch_end = local_start.replace(hour=lunch_end_hour, minute=0, second=0, microsecond=0)

    lunch_interval = TimeInterval(local_lunch_start, local_lunch_end)
    candidate_local_interval = TimeInterval(local_start, local_end)

    return candidate_local_interval.overlaps(lunch_interval)
