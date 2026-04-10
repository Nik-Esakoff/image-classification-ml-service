from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session

from dependencies import get_current_user, get_db
from schemas import PredictRequest, PredictResponse
from services import submit_prediction


router = APIRouter(prefix="/predict", tags=["predict"])


@router.post("", response_model=PredictResponse, status_code=status.HTTP_201_CREATED)
def create_prediction(
    payload: PredictRequest,
    db: Session = Depends(get_db),
    current_user=Depends(get_current_user),
):
    try:
        task = submit_prediction(
            session=db,
            user_id=current_user.id,
            model_code=payload.model_id,
            data=payload.data,
        )
        return task
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e),
        )
