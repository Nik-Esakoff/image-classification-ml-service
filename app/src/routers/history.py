from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session

from dependencies import get_db, get_current_user
from schemas import TaskHistoryResponse, TransactionHistoryResponse
from services import get_user_tasks_history, get_user_transactions_history, get_task_by_id


router = APIRouter(prefix="/history", tags=["history"])


@router.get("/tasks", response_model=list[TaskHistoryResponse])
def get_my_tasks(
    db: Session = Depends(get_db),
    current_user=Depends(get_current_user),
):
    return get_user_tasks_history(db, current_user.id)


@router.get("/transactions", response_model=list[TransactionHistoryResponse])
def get_my_transactions(
    db: Session = Depends(get_db),
    current_user=Depends(get_current_user),
):
    return get_user_transactions_history(db, current_user.id)


@router.get("/tasks/{task_id}", response_model=TaskHistoryResponse)
def get_task(
    task_id: int,
    db: Session = Depends(get_db),
    current_user=Depends(get_current_user),
):
    try:
        return get_task_by_id(db, task_id, current_user.id)
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(e)
        )
