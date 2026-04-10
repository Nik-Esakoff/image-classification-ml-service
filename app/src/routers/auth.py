from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session

from dependencies import get_db
from schemas import UserRegisterRequest, UserResponse
from services import create_user


router = APIRouter(prefix="/auth", tags=["auth"])


@router.post(
    "/register",
    response_model=UserResponse,
    status_code=status.HTTP_201_CREATED,
)
def register_user(payload: UserRegisterRequest, db: Session = Depends(get_db)):
    try:
        user = create_user(
            session=db,
            login=payload.login,
            password_hash=payload.password,
        )
        return user
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=str(e),
        )
