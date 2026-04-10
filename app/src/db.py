import os
from sqlalchemy import create_engine
from sqlalchemy.orm import DeclarativeBase, sessionmaker


class Base(DeclarativeBase):
    pass


DB_HOST = os.getenv("DATABASE_HOST", "database")
DB_NAME = os.getenv("DATABASE_NAME", "ml_service")
DB_PORT = os.getenv("DATABASE_PORT", "5432")
DB_USER = os.getenv("DATABASE_USER", "postgres")
DB_PASSWORD = os.getenv("DATABASE_PASSWORD", "secret")

DATABASE_URL = (
    f"postgresql+psycopg://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}"
)

engine = create_engine(DATABASE_URL, echo=True)

SessionLocal = sessionmaker(bind=engine)
