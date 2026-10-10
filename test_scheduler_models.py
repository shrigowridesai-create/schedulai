from datetime import datetime, date, timezone
from app.db import init_db, SessionLocal
from app.models.db_models import User, CalendarEvent, PriorityLevel
from app.models.schemas import SchedulingRequest
from app.core.intervals import TimeInterval
from app.core.constraint_engine import ConstraintEngine
from app.ml.model import MeetingPreferenceModel


def run_problem_statement_51_test():
    print("=" * 60)
    print("RUNNING PROBLEM STATEMENT 51 VALIDATION TEST")
    print("=" * 60)

    init_db()
    db = SessionLocal()

    # Clear existing test data
    db.query(CalendarEvent).delete()
    db.query(User).delete()
    db.commit()

    # 1. Create Manager (Organizer)
    manager = User(
        name="Sarah Manager",
        email="manager@company.com",
        timezone="UTC",
        work_start_hour=9,
        work_end_hour=17,
        lunch_start_hour=12,
        lunch_end_hour=13
    )
    db.add(manager)

    # 2. Create 5 Employees
    # Employee 1 & 2: Already have meetings at 10-11 AM
    emp1 = User(name="Alice (Busy 10-11)", email="alice@company.com", timezone="UTC", work_start_hour=9, work_end_hour=17, lunch_start_hour=12, lunch_end_hour=13)
    emp2 = User(name="Bob (Busy 10-11)", email="bob@company.com", timezone="UTC", work_start_hour=9, work_end_hour=17, lunch_start_hour=12, lunch_end_hour=13)
    # Employee 3: Prefers no meetings at lunch 12-1 PM
    emp3 = User(name="Charlie (Lunch 12-1)", email="charlie@company.com", timezone="UTC", work_start_hour=9, work_end_hour=17, lunch_start_hour=12, lunch_end_hour=13)
    # Employee 4: Works in a different timezone (e.g. America/New_York)
    emp4 = User(name="Diana (New York TZ)", email="diana@company.com", timezone="America/New_York", work_start_hour=9, work_end_hour=17, lunch_start_hour=12, lunch_end_hour=13)
    # Employee 5: Regular employee
    emp5 = User(name="Evan", email="evan@company.com", timezone="UTC", work_start_hour=9, work_end_hour=17, lunch_start_hour=12, lunch_end_hour=13)

    employees = [emp1, emp2, emp3, emp4, emp5]
    db.add_all(employees)
    db.commit()
    for e in [manager] + employees:
        db.refresh(e)

    # 3. Add existing calendar commitments for target date: 2026-10-15
    target_date = date(2026, 10, 15)

    # Conflict: Alice and Bob have meetings from 10:00 to 11:00 UTC
    event1 = CalendarEvent(
        user_id=emp1.id,
        title="Sprint Retrospective",
        start_time=datetime(2026, 10, 15, 10, 0, tzinfo=timezone.utc),
        end_time=datetime(2026, 10, 15, 11, 0, tzinfo=timezone.utc),
        source="internal"
    )
    event2 = CalendarEvent(
        user_id=emp2.id,
        title="Client Demo",
        start_time=datetime(2026, 10, 15, 10, 0, tzinfo=timezone.utc),
        end_time=datetime(2026, 10, 15, 11, 0, tzinfo=timezone.utc),
        source="internal"
    )
    db.add_all([event1, event2])
    db.commit()

    # 4. Prepare data for Constraint Engine
    attendees_data = []
    for emp in [manager] + employees:
        events = db.query(CalendarEvent).filter(CalendarEvent.user_id == emp.id).all()
        attendees_data.append({
            "id": emp.id,
            "name": emp.name,
            "email": emp.email,
            "timezone": emp.timezone,
            "work_start_hour": emp.work_start_hour,
            "work_end_hour": emp.work_end_hour,
            "lunch_start_hour": emp.lunch_start_hour,
            "lunch_end_hour": emp.lunch_end_hour,
            "is_required": True,
            "busy_intervals": [
                TimeInterval(e.start_time, e.end_time) for e in events
            ]
        })

    # 5. Run Constraint Engine with trained Scikit-learn model
    ml_model = MeetingPreferenceModel()
    print(f"ML Model Trained Status: {ml_model.is_trained()}")

    engine = ConstraintEngine(ml_model=ml_model)
    result = engine.evaluate_and_recommend(
        meeting_title="Manager Sync with 5 Employees",
        target_date=target_date,
        duration_minutes=60,
        organizer=attendees_data[0],
        participants=attendees_data[1:],
        priority="HIGH",
        search_start_hour=9,
        search_end_hour=17
    )

    print(f"\nTotal Candidate Slots Evaluated: {result.total_slots_evaluated}")
    print(f"Feasible Slots: {result.feasible_slots_count}")

    print("\n--- TOP RECOMMENDATION ---")
    if result.best_recommendation:
        b = result.best_recommendation
        start_fmt = b.start_time.strftime('%I:%M %p')
        end_fmt = b.end_time.strftime('%I:%M %p')
        print(f"Time: {start_fmt} - {end_fmt} UTC")
        print(f"Score: {b.preference_score:.3f}")
        print(f"Explanation: {b.explanation}")

    print("\n--- ALTERNATIVE RECOMMENDATIONS ---")
    for alt in result.alternative_recommendations:
        start_fmt = alt.start_time.strftime('%I:%M %p')
        end_fmt = alt.end_time.strftime('%I:%M %p')
        print(f"Rank #{alt.rank}: {start_fmt} - {end_fmt} UTC (Score: {alt.preference_score:.3f}) | {alt.explanation}")

    print("\n--- REJECTED SLOTS SUMMARY ---")
    for rej in result.rejected_slots_summary[:4]:
        start_fmt = rej.start_time.strftime('%I:%M %p')
        end_fmt = rej.end_time.strftime('%I:%M %p')
        print(f"Rejected: {start_fmt} - {end_fmt} UTC | Reasons: {'; '.join(rej.hard_conflicts)}")

    db.close()
    print("\n" + "=" * 60)
    print("TEST COMPLETED SUCCESSFULLY!")
    print("=" * 60)


if __name__ == "__main__":
    run_problem_statement_51_test()
