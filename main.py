from datetime import datetime, date
from typing import List
from fastapi import FastAPI, Depends, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy.orm import Session
import pytz

from .db import init_db, get_db
from .models.db_models import User, Meeting, MeetingParticipant, CalendarEvent, MeetingFeedback
from .models.schemas import (
    UserCreate, UserResponse,
    CalendarEventCreate, CalendarEventResponse,
    SchedulingRequest, SchedulingRecommendationResponse,
    FeedbackCreate, FeedbackResponse
)
from .core.intervals import TimeInterval
from .core.constraint_engine import ConstraintEngine
from .ml.model import MeetingPreferenceModel

app = FastAPI(
    title="AI-Based Meeting Conflict Predictor API",
    description="Smart scheduling API that identifies conflicts, learns participant preferences, and recommends optimal meeting times.",
    version="1.0.0"
)

# Enable CORS for React frontend (Vite default port 5173, etc.)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize Database tables and ML Engine
init_db()
ml_model = MeetingPreferenceModel()
constraint_engine = ConstraintEngine(ml_model=ml_model)


@app.get("/api/health")
def health_check():
    return {
        "status": "healthy",
        "ml_model_loaded": ml_model.is_trained(),
        "timestamp": datetime.now().isoformat()
    }


# ===================== USER ENDPOINTS =====================

@app.post("/api/users/", response_model=UserResponse)
def create_user(user_in: UserCreate, db: Session = Depends(get_db)):
    existing = db.query(User).filter(User.email == user_in.email).first()
    if existing:
        raise HTTPException(status_code=400, detail="User with this email already exists.")
    user = User(**user_in.model_dump())
    db.add(user)
    db.commit()
    db.refresh(user)
    return user


@app.get("/api/users/", response_model=List[UserResponse])
def list_users(db: Session = Depends(get_db)):
    return db.query(User).all()


@app.get("/api/users/{user_id}", response_model=UserResponse)
def get_user(user_id: int, db: Session = Depends(get_db)):
    user = db.query(User).filter(User.id == user_id).first()
    if not user:
        raise HTTPException(status_code=404, detail="User not found.")
    return user


# ===================== CALENDAR EVENT ENDPOINTS =====================

@app.post("/api/events/", response_model=CalendarEventResponse)
def add_calendar_event(event_in: CalendarEventCreate, db: Session = Depends(get_db)):
    user = db.query(User).filter(User.id == event_in.user_id).first()
    if not user:
        raise HTTPException(status_code=404, detail="User not found.")

    if event_in.start_time >= event_in.end_time:
        raise HTTPException(status_code=400, detail="start_time must be strictly before end_time.")

    # Ensure UTC timezone awareness
    start_utc = event_in.start_time if event_in.start_time.tzinfo else pytz.UTC.localize(event_in.start_time)
    end_utc = event_in.end_time if event_in.end_time.tzinfo else pytz.UTC.localize(event_in.end_time)

    event = CalendarEvent(
        user_id=event_in.user_id,
        title=event_in.title,
        start_time=start_utc,
        end_time=end_utc,
        is_all_day=event_in.is_all_day,
        source=event_in.source,
        external_event_id=event_in.external_event_id
    )
    db.add(event)
    db.commit()
    db.refresh(event)
    return event


@app.get("/api/events/user/{user_id}", response_model=List[CalendarEventResponse])
def get_user_events(user_id: int, db: Session = Depends(get_db)):
    return db.query(CalendarEvent).filter(CalendarEvent.user_id == user_id).all()


# ===================== SCHEDULING RECOMMENDATION =====================

@app.post("/api/schedule/recommend", response_model=SchedulingRecommendationResponse)
def recommend_meeting_times(req: SchedulingRequest, db: Session = Depends(get_db)):
    organizer = db.query(User).filter(User.id == req.organizer_id).first()
    if not organizer:
        raise HTTPException(status_code=404, detail="Organizer not found.")

    participants = db.query(User).filter(User.id.in_(req.participant_ids)).all()
    if not participants:
        raise HTTPException(status_code=400, detail="No valid participants found.")

    try:
        target_date = date.fromisoformat(req.target_date)
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid target_date format. Use YYYY-MM-DD.")

    required_set = set(req.required_participant_ids or req.participant_ids)

    # Prepare organizer data
    org_events = db.query(CalendarEvent).filter(CalendarEvent.user_id == organizer.id).all()
    org_data = {
        "id": organizer.id,
        "name": organizer.name,
        "email": organizer.email,
        "timezone": organizer.timezone,
        "work_start_hour": organizer.work_start_hour,
        "work_end_hour": organizer.work_end_hour,
        "lunch_start_hour": organizer.lunch_start_hour,
        "lunch_end_hour": organizer.lunch_end_hour,
        "is_required": True,
        "busy_intervals": [
            TimeInterval(
                start=e.start_time if e.start_time.tzinfo else pytz.UTC.localize(e.start_time),
                end=e.end_time if e.end_time.tzinfo else pytz.UTC.localize(e.end_time)
            ) for e in org_events
        ]
    }

    # Prepare participant data
    participants_data = []
    for p in participants:
        p_events = db.query(CalendarEvent).filter(CalendarEvent.user_id == p.id).all()
        participants_data.append({
            "id": p.id,
            "name": p.name,
            "email": p.email,
            "timezone": p.timezone,
            "work_start_hour": p.work_start_hour,
            "work_end_hour": p.work_end_hour,
            "lunch_start_hour": p.lunch_start_hour,
            "lunch_end_hour": p.lunch_end_hour,
            "is_required": p.id in required_set,
            "busy_intervals": [
                TimeInterval(
                    start=e.start_time if e.start_time.tzinfo else pytz.UTC.localize(e.start_time),
                    end=e.end_time if e.end_time.tzinfo else pytz.UTC.localize(e.end_time)
                ) for e in p_events
            ]
        })

    # Execute Constraint Engine + ML Model
    result = constraint_engine.evaluate_and_recommend(
        meeting_title="Meeting Sync",
        target_date=target_date,
        duration_minutes=req.duration_minutes,
        organizer=org_data,
        participants=participants_data,
        priority=req.priority.value,
        search_start_hour=req.search_window_start_hour,
        search_end_hour=req.search_window_end_hour
    )

    return result


# ===================== FEEDBACK / CONTINUOUS LEARNING =====================

@app.post("/api/feedback/", response_model=FeedbackResponse)
def submit_feedback(fb_in: FeedbackCreate, db: Session = Depends(get_db)):
    feedback = MeetingFeedback(**fb_in.model_dump())
    db.add(feedback)
    db.commit()
    db.refresh(feedback)
    return feedback
