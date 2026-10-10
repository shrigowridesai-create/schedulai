from datetime import datetime, timezone
import enum
from typing import List, Optional

from sqlalchemy import (
    Column, Integer, String, Boolean, DateTime, ForeignKey, Enum, Text, Float
)
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship


class Base(DeclarativeBase):
    pass


class PriorityLevel(str, enum.Enum):
    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"
    CRITICAL = "CRITICAL"


class MeetingStatus(str, enum.Enum):
    PROPOSED = "PROPOSED"
    CONFIRMED = "CONFIRMED"
    CANCELLED = "CANCELLED"


class ParticipantStatus(str, enum.Enum):
    PENDING = "PENDING"
    ACCEPTED = "ACCEPTED"
    DECLINED = "DECLINED"


class User(Base):
    __tablename__ = "users"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True)
    name: Mapped[str] = mapped_column(String(100), nullable=False)
    email: Mapped[str] = mapped_column(String(255), unique=True, index=True, nullable=False)
    timezone: Mapped[str] = mapped_column(String(50), default="UTC", nullable=False)
    
    # Working hours (Local time, 24-hr format)
    work_start_hour: Mapped[int] = mapped_column(Integer, default=9, nullable=False)
    work_end_hour: Mapped[int] = mapped_column(Integer, default=17, nullable=False)
    
    # Lunch break preferences
    lunch_start_hour: Mapped[int] = mapped_column(Integer, default=12, nullable=False)
    lunch_end_hour: Mapped[int] = mapped_column(Integer, default=13, nullable=False)
    
    # Psychological / scheduling comfort constraints
    preferred_buffer_minutes: Mapped[int] = mapped_column(Integer, default=15, nullable=False)
    max_consecutive_meetings: Mapped[int] = mapped_column(Integer, default=3, nullable=False)
    
    created_at: Mapped[datetime] = mapped_column(
        DateTime, default=lambda: datetime.now(timezone.utc), nullable=False
    )

    # Relationships
    calendar_events: Mapped[List["CalendarEvent"]] = relationship("CalendarEvent", back_populates="user", cascade="all, delete-orphan")
    organized_meetings: Mapped[List["Meeting"]] = relationship("Meeting", back_populates="organizer")
    participations: Mapped[List["MeetingParticipant"]] = relationship("MeetingParticipant", back_populates="user")
    feedbacks: Mapped[List["MeetingFeedback"]] = relationship("MeetingFeedback", back_populates="user")


class Meeting(Base):
    __tablename__ = "meetings"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True)
    title: Mapped[str] = mapped_column(String(200), nullable=False)
    description: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    organizer_id: Mapped[int] = mapped_column(Integer, ForeignKey("users.id"), nullable=False)
    
    duration_minutes: Mapped[int] = mapped_column(Integer, default=60, nullable=False)
    priority: Mapped[PriorityLevel] = mapped_column(
        Enum(PriorityLevel), default=PriorityLevel.MEDIUM, nullable=False
    )
    status: Mapped[MeetingStatus] = mapped_column(
        Enum(MeetingStatus), default=MeetingStatus.PROPOSED, nullable=False
    )
    
    # Final confirmed slot
    confirmed_start_time: Mapped[Optional[datetime]] = mapped_column(DateTime, nullable=True)
    confirmed_end_time: Mapped[Optional[datetime]] = mapped_column(DateTime, nullable=True)
    
    created_at: Mapped[datetime] = mapped_column(
        DateTime, default=lambda: datetime.now(timezone.utc), nullable=False
    )

    # Relationships
    organizer: Mapped["User"] = relationship("User", back_populates="organized_meetings")
    participants: Mapped[List["MeetingParticipant"]] = relationship("MeetingParticipant", back_populates="meeting", cascade="all, delete-orphan")
    feedbacks: Mapped[List["MeetingFeedback"]] = relationship("MeetingFeedback", back_populates="meeting")


class MeetingParticipant(Base):
    __tablename__ = "meeting_participants"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True)
    meeting_id: Mapped[int] = mapped_column(Integer, ForeignKey("meetings.id"), nullable=False)
    user_id: Mapped[int] = mapped_column(Integer, ForeignKey("users.id"), nullable=False)
    
    is_required: Mapped[bool] = mapped_column(Boolean, default=True, nullable=False)
    status: Mapped[ParticipantStatus] = mapped_column(
        Enum(ParticipantStatus), default=ParticipantStatus.PENDING, nullable=False
    )

    # Relationships
    meeting: Mapped["Meeting"] = relationship("Meeting", back_populates="participants")
    user: Mapped["User"] = relationship("User", back_populates="participations")


class CalendarEvent(Base):
    """
    Represents existing commitments/busy blocks on a participant's calendar.
    Can be entered manually or synced from Google Calendar / Microsoft Outlook.
    """
    __tablename__ = "calendar_events"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True)
    user_id: Mapped[int] = mapped_column(Integer, ForeignKey("users.id"), nullable=False)
    title: Mapped[str] = mapped_column(String(200), nullable=False)
    
    # UTC timestamps for consistency across timezones
    start_time: Mapped[datetime] = mapped_column(DateTime, nullable=False, index=True)
    end_time: Mapped[datetime] = mapped_column(DateTime, nullable=False, index=True)
    
    is_all_day: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False)
    source: Mapped[str] = mapped_column(String(50), default="internal", nullable=False) # 'internal', 'google_calendar'
    external_event_id: Mapped[Optional[str]] = mapped_column(String(255), nullable=True)

    # Relationships
    user: Mapped["User"] = relationship("User", back_populates="calendar_events")


class MeetingFeedback(Base):
    """
    Stores historical scheduling outcomes & user feedback to retrain the ML model.
    """
    __tablename__ = "meeting_feedbacks"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True)
    user_id: Mapped[int] = mapped_column(Integer, ForeignKey("users.id"), nullable=False)
    meeting_id: Mapped[int] = mapped_column(Integer, ForeignKey("meetings.id"), nullable=False)
    
    slot_start: Mapped[datetime] = mapped_column(DateTime, nullable=False)
    slot_end: Mapped[datetime] = mapped_column(DateTime, nullable=False)
    
    # ML Ground Truth Target
    accepted: Mapped[bool] = mapped_column(Boolean, nullable=False)  # 1 = accepted/liked, 0 = rejected/rescheduled
    rating: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)  # 1 to 5 scale
    
    # Feedback reasons for explanation auditing
    reason: Mapped[Optional[str]] = mapped_column(String(255), nullable=True) 
    
    created_at: Mapped[datetime] = mapped_column(
        DateTime, default=lambda: datetime.now(timezone.utc), nullable=False
    )

    # Relationships
    user: Mapped["User"] = relationship("User", back_populates="feedbacks")
    meeting: Mapped["Meeting"] = relationship("Meeting", back_populates="feedbacks")
