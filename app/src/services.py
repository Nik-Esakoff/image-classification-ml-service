from sqlalchemy import select
from sqlalchemy.orm import Session

from models import MLModel, MLTask, TaskStatus, Transaction, TransactionType, User, UserRole


def create_user(
    session: Session,
    login: str,
    password_hash: str,
    role: UserRole = UserRole.USER,
    balance: int = 100,
) -> User:
    """Создаёт пользователя, если логин ещё не занят."""
    existing_user = session.scalar(
        select(User).where(User.login == login)
    )
    if existing_user is not None:
        raise ValueError(f"Пользователь с логином '{login}' уже существует")

    user = User(
        login=login,
        password_hash=password_hash,
        role=role,
        balance=balance,
    )
    session.add(user)
    session.commit()
    session.refresh(user)
    return user


def get_user_by_id(session: Session, user_id: int) -> User | None:
    """Возвращает пользователя по id или None."""
    return session.get(User, user_id)


def get_user_by_login(session: Session, login: str) -> User | None:
    """Возвращает пользователя по логину или None."""
    return session.scalar(
        select(User).where(User.login == login)
    )


def deposit_balance(session: Session, user_id: int, amount: int) -> Transaction:
    """Пополняет баланс пользователя и сохраняет транзакцию."""
    if amount <= 0:
        raise ValueError("Сумма пополнения должна быть больше 0")

    user = session.get(User, user_id)
    if user is None:
        raise ValueError("Пользователь не найден")

    user.balance += amount

    transaction = Transaction(
        user_id=user.id,
        amount=amount,
        transaction_type=TransactionType.CREDIT,
    )
    session.add(transaction)
    session.commit()
    session.refresh(transaction)
    return transaction


def withdraw_balance(
    session: Session,
    user_id: int,
    amount: int,
    task_id: int | None = None,
) -> Transaction:
    """Списывает баланс пользователя и сохраняет транзакцию."""
    if amount <= 0:
        raise ValueError("Сумма списания должна быть больше 0")

    user = session.get(User, user_id)
    if user is None:
        raise ValueError("Пользователь не найден")

    if task_id is not None:
        task = session.get(MLTask, task_id)
        if task is None:
            raise ValueError("ML-задача не найдена")
        if task.user_id != user.id:
            raise ValueError("Эта задача не принадлежит пользователю")

    if user.balance < amount:
        raise ValueError("Недостаточно средств на балансе")

    user.balance -= amount

    transaction = Transaction(
        user_id=user.id,
        task_id=task_id,
        amount=amount,
        transaction_type=TransactionType.DEBIT,
    )
    session.add(transaction)
    session.commit()
    session.refresh(transaction)
    return transaction


def create_task(session: Session, user_id: int, model_id: int, data: str) -> MLTask:
    """Создаёт ML-задачу пользователя."""
    user = session.get(User, user_id)
    if user is None:
        raise ValueError("Пользователь не найден")

    model = session.get(MLModel, model_id)
    if model is None:
        raise ValueError("ML-модель не найдена")

    task = MLTask(
        user_id=user.id,
        model_id=model.id,
        data=data,
        status=TaskStatus.CREATED,
    )
    session.add(task)
    session.commit()
    session.refresh(task)
    return task


def get_user_transactions_history(session: Session, user_id: int) -> list[Transaction]:
    """Возвращает историю транзакций пользователя, отсортированную по дате."""
    user = session.get(User, user_id)
    if user is None:
        raise ValueError("Пользователь не найден")

    return list(
        session.scalars(
            select(Transaction)
            .where(Transaction.user_id == user_id)
            .order_by(Transaction.created_at.desc())
        ).all()
    )


def get_user_tasks_history(session: Session, user_id: int) -> list[MLTask]:
    """Возвращает историю ML-задач пользователя, отсортированную по дате."""
    user = session.get(User, user_id)
    if user is None:
        raise ValueError("Пользователь не найден")

    return list(
        session.scalars(
            select(MLTask)
            .where(MLTask.user_id == user_id)
            .order_by(MLTask.created_at.desc())
        ).all()
    )