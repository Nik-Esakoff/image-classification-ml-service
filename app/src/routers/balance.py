from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session

from dependencies import get_current_user, get_db
from schemas import BalanceDepositRequest, BalanceResponse
from services import deposit_balance, get_user_by_id


router = APIRouter(prefix="/balance", tags=["balance"])


@router.get("/", response_model=BalanceResponse)
def get_balance(current_user=Depends(get_current_user)):
    return BalanceResponse(balance=current_user.balance)


@router.post("/deposit", response_model=BalanceResponse)
def deposit_user_balance(
    payload: BalanceDepositRequest,
    db: Session = Depends(get_db),
    current_user=Depends(get_current_user),
):
    deposit_balance(db, current_user.id, payload.amount)
    updated_user = get_user_by_id(db, current_user.id)
    return BalanceResponse(balance=updated_user.balance)
