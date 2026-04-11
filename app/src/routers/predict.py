from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session

from dependencies import get_current_user, get_db
from schemas import PredictAcceptedResponse, PredictRequest
from services import enqueue_prediction


router = APIRouter(prefix="/predict", tags=["predict"])


def map_prediction_error_to_status(message: str) -> int:
    if message in {"Пользователь не найден", "ML-модель не найдена"}:
        return status.HTTP_404_NOT_FOUND
    if message == "Недостаточно средств на балансе":
        return status.HTTP_409_CONFLICT
    return status.HTTP_400_BAD_REQUEST


@router.post(
    "",
    response_model=PredictAcceptedResponse,
    status_code=status.HTTP_202_ACCEPTED,
)
def create_prediction(
    payload: PredictRequest,
    db: Session = Depends(get_db),
    current_user=Depends(get_current_user),
):
    try:
        task = enqueue_prediction(
            session=db,
            user_id=current_user.id,
            model_code=payload.model_id,
            data=payload.data,
        )

        return PredictAcceptedResponse(
            task_id=task.id,
            status=task.status,
            message="Task accepted and sent to RabbitMQ queue",
        )

    except ValueError as e:
        raise HTTPException(
            status_code=map_prediction_error_to_status(str(e)),
            detail=str(e),
        )

    except RuntimeError as e:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=str(e),
        )
