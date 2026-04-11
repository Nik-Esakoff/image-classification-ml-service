from pydantic import BaseModel, ConfigDict, Field
from models import UserRole

from datetime import datetime
from models import TaskStatus, TransactionType


class UserRegisterRequest(BaseModel):
    login: str = Field(min_length=3, max_length=50)
    password: str = Field(min_length=4, max_length=100)


class UserResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    login: str
    role: UserRole
    balance: int


class BalanceDepositRequest(BaseModel):
    amount: int = Field(gt=0)


class BalanceResponse(BaseModel):
    balance: int


class TaskHistoryResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    model_id: int
    data: str
    status: TaskStatus
    created_at: datetime
    result: str | None


class TransactionHistoryResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    task_id: int | None
    amount: int
    transaction_type: TransactionType
    created_at: datetime


class PredictRequest(BaseModel):
    model_id: str = Field(min_length=1, max_length=100)
    data: str = Field(min_length=1, max_length=255)


class PredictResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    model_id: int
    data: str
    status: TaskStatus
    created_at: datetime
    result: str | None


class PredictAcceptedResponse(BaseModel):
    task_id: int
    status: TaskStatus
    message: str
