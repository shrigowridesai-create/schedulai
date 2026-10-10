import os
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from .models.db_models import Base

# Supports PostgreSQL (via DATABASE_URL env) with automatic SQLite fallback for local development
DEFAULT_SQLITE_PATH = os.path.join(os.path.expanduser("~"), "ai-meeting-scheduler", "scheduler.db")
DATABASE_URL = os.getenv("DATABASE_URL", f"sqlite:///{DEFAULT_SQLITE_PATH.replace(os.sep, '/')}")

# SQLite requires check_same_thread=False
connect_args = {"check_same_thread": False} if DATABASE_URL.startswith("sqlite") else {}

engine = create_engine(DATABASE_URL, connect_args=connect_args)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


def init_db():
    """Creates all tables if they don't already exist."""
    Base.metadata.create_all(bind=engine)


def get_db():
    """FastAPI dependency for database session."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
