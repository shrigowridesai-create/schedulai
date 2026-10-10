from datetime import datetime
from typing import List, Optional
from pydantic import BaseModel, EmailStr, Field

from .db_models import PriorityLevel, MeetingStatus, ParticipantStatus


# ===================== USER SCHEMAS =====================

class UserBase(BaseModel):
    name: str = Field(..., example="Alice Johnson")
    email: EmailStr = Field(..., example="alice@company.com")
    timezone: str = Field(default="UTC", example="America/New_York")
    work_start_hour: int = Field(default=9, ge=0, le=23, example=9)
    work_end_hour: int = Field(default=17, ge=0, le=23, example=17)
    lunch_start_hour: int = Field(default=12, ge=0, le=23, example=12)
    lunch_end_hour: int = Field(default=13, ge=0, le=23, example=13)
    preferred_buffer_minutes: int = Field(default=15, ge=0, example=15)
    max_consecutive_meetings: int = Field(default=3, ge=1, example=3)


class UserCreate(UserBase):
    pass


class UserResponse(UserBase):
    id: int
    created_at: datetime

    class Config:
        from_attributes = True


# ===================== CALENDAR EVENT SCHEMAS =====================

class CalendarEventBase(BaseModel):
    title: str = Field(..., example="Project Sprint Planning")
    start_time: datetime
    end_time: datetime
    is_all_day: bool = False
    source: str = "internal"
    external_event_id: Optional[str] = None


class CalendarEventCreate(CalendarEventBase):
    user_id: int


class CalendarEventResponse(CalendarEventBase):
    id: int
    user_id: int

    class Config:
        from_attributes = True


# ===================== PARTICIPANT SCHEMAS =====================

class ParticipantInput(BaseModel):
    user_id: int
    is_required: bool = True


class ParticipantResponse(BaseModel):
    user_id: int
    name: str
    email: str
    is_required: bool
    status: ParticipantStatus

    class Config:
        from_attributes = True


# ===================== MEETING SCHEMAS =====================

class MeetingCreate(BaseModel):
    title: str = Field(..., example="Strategy Sync")
    description: Optional[str] = None
    organizer_id: int
    duration_minutes: int = Field(default=60, gt=0, example=60)
    priority: PriorityLevel = PriorityLevel.MEDIUM
    participants: List[ParticipantInput]


class MeetingResponse(BaseModel):
    id: int
    title: str
    description: Optional[str]
    organizer_id: int
    duration_minutes: int
    priority: PriorityLevel
    status: MeetingStatus
    confirmed_start_time: Optional[datetime]
    confirmed_end_time: Optional[datetime]
    created_at: datetime
    participants: List[ParticipantResponse]

    class Config:
        from_attributes = True


# ===================== SCHEDULING & RECOMMENDATION SCHEMAS =====================

class SchedulingRequest(BaseModel):
    organizer_id: int
    participant_ids: List[int]
    required_participant_ids: Optional[List[int]] = None
    duration_minutes: int = Field(default=60, gt=0)
    target_date: str = Field(..., example="2026-10-15") # YYYY-MM-DD
    priority: PriorityLevel = PriorityLevel.MEDIUM
    search_window_start_hour: int = Field(default=8, ge=0, le=23)
    search_window_end_hour: int = Field(default=18, ge=0, le=23)


class CandidateSlotResponse(BaseModel):
    start_time: datetime
    end_time: datetime
    is_feasible: bool
    hard_conflicts: List[str] = []
    preference_score: float = Field(..., description="ML-predicted suitability score (0.0 to 1.0)")
    rank: int = Field(..., description="1 = best recommended slot")
    explanation: str = Field(..., description="Human-readable breakdown of decision and trade-offs")
    compromises: List[str] = []


class SchedulingRecommendationResponse(BaseModel):
    meeting_title: Optional[str] = None
    target_date: str
    duration_minutes: int
    total_slots_evaluated: int
    feasible_slots_count: int
    best_recommendation: Optional[CandidateSlotResponse]
    alternative_recommendations: List[CandidateSlotResponse] = []
    rejected_slots_summary: List[CandidateSlotResponse] = []


# ===================== FEEDBACK SCHEMAS =====================

class FeedbackCreate(BaseModel):
    user_id: int
    meeting_id: int
    slot_start: datetime
    slot_end: datetime
    accepted: bool
    rating: Optional[int] = Field(None, ge=1, le=5)
    reason: Optional[str] = None


class FeedbackResponse(FeedbackCreate):
    id: int
    created_at: datetime

    class Config:
        from_attributes = True
