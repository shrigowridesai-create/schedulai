from datetime import datetime, date, time, timedelta, timezone
from typing import List, Dict, Any, Optional, Tuple
import pytz

from .intervals import (
    TimeInterval, is_within_working_hours, overlaps_lunch, calculate_buffer_minutes
)
from ..ml.feature_extractor import extract_slot_features
from ..ml.model import MeetingPreferenceModel
from ..models.schemas import CandidateSlotResponse, SchedulingRecommendationResponse


class ConstraintEngine:
    """
    Core engine combining:
    1. Interval-based Hard Constraint Engine (Overlaps, working hours, required attendees)
    2. Scikit-learn ML Preference Model (Ranks viable alternative slots)
    3. Explainability Engine (Generates human-readable reasons and compromises)
    """

    def __init__(self, ml_model: Optional[MeetingPreferenceModel] = None):
        self.ml_model = ml_model or MeetingPreferenceModel()

    def evaluate_and_recommend(
        self,
        meeting_title: str,
        target_date: date,
        duration_minutes: int,
        organizer: Dict[str, Any],
        participants: List[Dict[str, Any]],
        priority: str = "MEDIUM",
        search_start_hour: int = 8,
        search_end_hour: int = 18,
        step_minutes: int = 30
    ) -> SchedulingRecommendationResponse:
        """
        Main entry point for scheduling recommendation.
        """
        all_attendees = [organizer] + [p for p in participants if p["id"] != organizer["id"]]
        
        # 1. Generate candidate intervals on target date
        candidate_intervals = self._generate_intervals(
            target_date=target_date,
            duration_minutes=duration_minutes,
            start_hour=search_start_hour,
            end_hour=search_end_hour,
            step_minutes=step_minutes
        )

        feasible_slots: List[CandidateSlotResponse] = []
        rejected_slots: List[CandidateSlotResponse] = []

        # 2. Evaluate each slot
        for interval in candidate_intervals:
            slot_eval = self._evaluate_slot(
                interval=interval,
                attendees=all_attendees,
                duration_minutes=duration_minutes,
                priority=priority
            )
            if slot_eval.is_feasible:
                feasible_slots.append(slot_eval)
            else:
                rejected_slots.append(slot_eval)

        # 3. Sort feasible slots by ML preference score (highest first)
        feasible_slots.sort(key=lambda s: s.preference_score, reverse=True)

        # 4. Assign ranks and build recommendations
        ranked_feasible: List[CandidateSlotResponse] = []
        for rank_idx, slot in enumerate(feasible_slots, start=1):
            ranked_slot = CandidateSlotResponse(
                start_time=slot.start_time,
                end_time=slot.end_time,
                is_feasible=True,
                hard_conflicts=[],
                preference_score=slot.preference_score,
                rank=rank_idx,
                explanation=self._build_recommendation_explanation(rank_idx, slot),
                compromises=slot.compromises
            )
            ranked_feasible.append(ranked_slot)

        best_rec = ranked_feasible[0] if ranked_feasible else None
        alternatives = ranked_feasible[1:5] if len(ranked_feasible) > 1 else []

        return SchedulingRecommendationResponse(
            meeting_title=meeting_title,
            target_date=target_date.isoformat(),
            duration_minutes=duration_minutes,
            total_slots_evaluated=len(candidate_intervals),
            feasible_slots_count=len(ranked_feasible),
            best_recommendation=best_rec,
            alternative_recommendations=alternatives,
            rejected_slots_summary=rejected_slots[:6] # Top rejected samples for audit
        )

    def _generate_intervals(
        self,
        target_date: date,
        duration_minutes: int,
        start_hour: int,
        end_hour: int,
        step_minutes: int
    ) -> List[TimeInterval]:
        """Generates candidate time intervals across the search window in UTC."""
        intervals = []
        curr_dt = datetime.combine(target_date, time(hour=start_hour, minute=0), tzinfo=timezone.utc)
        end_limit = datetime.combine(target_date, time(hour=end_hour, minute=0), tzinfo=timezone.utc)

        duration_td = timedelta(minutes=duration_minutes)
        step_td = timedelta(minutes=step_minutes)

        while curr_dt + duration_td <= end_limit:
            interval_end = curr_dt + duration_td
            intervals.append(TimeInterval(start=curr_dt, end=interval_end))
            curr_dt += step_td

        return intervals

    def _evaluate_slot(
        self,
        interval: TimeInterval,
        attendees: List[Dict[str, Any]],
        duration_minutes: int,
        priority: str
    ) -> CandidateSlotResponse:
        """Evaluates hard constraints, extracts ML features, and builds explanations."""
        hard_conflicts: List[str] = []
        compromises: List[str] = []

        # A. Check Hard Constraints
        for person in attendees:
            is_req = person.get("is_required", True)
            tz_str = person.get("timezone", "UTC")
            name = person.get("name", f"User #{person.get('id')}")

            # Check working hours
            within_work, work_msg = is_within_working_hours(
                interval,
                tz_str,
                person.get("work_start_hour", 9),
                person.get("work_end_hour", 17)
            )
            if not within_work:
                if is_req:
                    hard_conflicts.append(f"{name}: {work_msg}")
                else:
                    compromises.append(f"Optional attendee {name}: {work_msg}")

            # Check busy calendar overlaps
            busy_list = person.get("busy_intervals", [])
            for b in busy_list:
                if interval.overlaps(b):
                    overlap_info = f"Conflict for {name} with existing event ({b.start.strftime('%H:%M')}-{b.end.strftime('%H:%M')} UTC)"
                    if is_req:
                        hard_conflicts.append(overlap_info)
                    else:
                        compromises.append(f"Optional attendee: {overlap_info}")

        # B. Feasibility status
        is_feasible = len(hard_conflicts) == 0

        # C. Soft constraint & compromise checks
        for person in attendees:
            name = person.get("name", "Participant")
            tz_str = person.get("timezone", "UTC")
            l_start = person.get("lunch_start_hour", 12)
            l_end = person.get("lunch_end_hour", 13)

            if overlaps_lunch(interval, tz_str, l_start, l_end):
                compromises.append(f"Overlaps with preferred lunch hour for {name} ({l_start}:00-{l_end}:00 local)")

            buf_before, buf_after, consec = calculate_buffer_minutes(interval, person.get("busy_intervals", []))
            if consec >= 2:
                compromises.append(f"High meeting fatigue for {name} ({consec} consecutive meetings)")
            elif buf_before < 10.0:
                compromises.append(f"Back-to-back schedule for {name} ({int(buf_before)} min buffer before)")

        # D. ML Scoring
        pref_score = 0.0
        if is_feasible:
            features = extract_slot_features(
                candidate=interval,
                participants_info=attendees,
                duration_minutes=duration_minutes,
                priority=priority
            )
            pref_score = self.ml_model.predict_score(features)

        # E. Human explanation
        if not is_feasible:
            explanation = f"Rejected: {'; '.join(hard_conflicts[:2])}"
        else:
            explanation = self._build_base_explanation(pref_score, compromises)

        return CandidateSlotResponse(
            start_time=interval.start,
            end_time=interval.end,
            is_feasible=is_feasible,
            hard_conflicts=hard_conflicts,
            preference_score=pref_score,
            rank=0,
            explanation=explanation,
            compromises=compromises
        )

    def _build_base_explanation(self, score: float, compromises: List[str]) -> str:
        if not compromises:
            return f"Excellent slot (Score: {score:.2f}). All participants available, healthy buffers, outside lunch."
        return f"Feasible with compromises (Score: {score:.2f}). {compromises[0]}."

    def _build_recommendation_explanation(self, rank: int, slot: CandidateSlotResponse) -> str:
        time_str = f"{slot.start_time.strftime('%I:%M %p')} - {slot.end_time.strftime('%I:%M %p')} UTC"
        if rank == 1:
            if not slot.compromises:
                return f"Top Recommendation ({time_str}): All required participants free. Zero calendar conflicts, no lunch overlap, healthy rest buffer."
            return f"Top Recommendation ({time_str}): Best available compromise across team calendars (Score: {slot.preference_score:.2f})."
        else:
            if slot.compromises:
                return f"Rank #{rank} Alternative ({time_str}): Viable, but note: {slot.compromises[0]}."
            return f"Rank #{rank} Alternative ({time_str}): Free slot with slightly lower preference rating."
