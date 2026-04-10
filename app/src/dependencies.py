from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPBasic, HTTPBasicCredentials
from sqlalchemy.orm import Session

from db import SessionLocal
from services import get_user_by_login

security = HTTPBasic()


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def get_current_user(
        credentials: HTTPBasicCredentials = Depends(security),
        db: Session = Depends(get_db)
):
    user = get_user_by_login(db, credentials.username)

    if user is None or credentials.password != user.password_hash:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid login or password",
            headers={"WWW-Authenticate": "Basic"},
        )
    return user
