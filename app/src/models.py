from datetime import datetime
from enum import Enum
from zoneinfo import ZoneInfo

from sqlalchemy import Enum as SqlEnum, String, ForeignKey
from sqlalchemy.orm import Mapped, mapped_column, relationship

from db import Base


class TaskStatus(Enum):
    """Возможные статусы выполнения ML-задачи."""
    CREATED = "created"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"


class UserRole(Enum):
    """Возможные роли пользователя в системе."""
    ADMIN = "admin"
    USER = "user"


class TransactionType(Enum):
    """Типы операций по балансу."""
    CREDIT = "credit"
    DEBIT = "debit"


class User(Base):
    """ORM-модель пользователя."""

    __tablename__ = "users"

    id: Mapped[int] = mapped_column(primary_key=True)
    login: Mapped[str] = mapped_column(String(50), unique=True, nullable=False)
    password_hash: Mapped[str] = mapped_column(String(255), nullable=False)
    role: Mapped[UserRole] = mapped_column(SqlEnum(UserRole), nullable=False)
    balance: Mapped[int] = mapped_column(default=100, nullable=False)

    tasks: Mapped[list["MLTask"]] = relationship(back_populates="user")
    transactions: Mapped[list["Transaction"]
                         ] = relationship(back_populates="user")


class MLTask(Base):
    """ORM-модель ML-задачи."""

    __tablename__ = "ml_tasks"

    id: Mapped[int] = mapped_column(primary_key=True)

    user_id: Mapped[int] = mapped_column(
        ForeignKey("users.id"), nullable=False)
    model_id: Mapped[int] = mapped_column(
        ForeignKey("ml_models.id"), nullable=False)

    data: Mapped[str] = mapped_column(String(255), nullable=False)
    status: Mapped[TaskStatus] = mapped_column(
        SqlEnum(TaskStatus),
        default=TaskStatus.CREATED,
        nullable=False,
    )
    created_at: Mapped[datetime] = mapped_column(
        default=lambda: datetime.now(ZoneInfo("Europe/Moscow")),
        nullable=False,
    )
    result: Mapped[str | None] = mapped_column(String(100), nullable=True)

    user: Mapped["User"] = relationship(back_populates="tasks")
    model: Mapped["MLModel"] = relationship(back_populates="tasks")
    transactions: Mapped[list["Transaction"]
                         ] = relationship(back_populates="task")


class Transaction(Base):
    """ORM-модель транзакции"""

    __tablename__ = "transactions"

    id: Mapped[int] = mapped_column(primary_key=True)
    user_id: Mapped[int] = mapped_column(
        ForeignKey("users.id"), nullable=False)
    task_id: Mapped[int | None] = mapped_column(
        ForeignKey("ml_tasks.id"), nullable=True)
    amount: Mapped[int] = mapped_column(nullable=False)
    transaction_type: Mapped[TransactionType] = mapped_column(
        SqlEnum(TransactionType), nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        default=lambda: datetime.now(ZoneInfo("Europe/Moscow")),
        nullable=False,
    )

    user: Mapped["User"] = relationship(back_populates="transactions")
    task: Mapped["MLTask"] = relationship(back_populates="transactions")


class MLModel(Base):
    """ORM-модель ML-модели."""

    __tablename__ = "ml_models"

    id: Mapped[int] = mapped_column(primary_key=True)
    model_id: Mapped[str] = mapped_column(
        String(100), unique=True, nullable=False)
    description: Mapped[str] = mapped_column(String(255), nullable=False)
    prediction_price: Mapped[int] = mapped_column(nullable=False)

    tasks: Mapped[list["MLTask"]] = relationship(back_populates="model")
