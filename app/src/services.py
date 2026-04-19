from sqlalchemy import select
from sqlalchemy.orm import Session

from models import MLModel, MLTask, TaskStatus, Transaction, TransactionType, User, UserRole
from datetime import datetime

from broker import publish_prediction_task


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


def get_task_by_id(session: Session, task_id: int, user_id: int) -> MLTask:
    """Возвращает ML-задачу по id"""
    task = session.scalar(
        select(MLTask)
        .where(MLTask.id == task_id)
    )
    if task is None:
        raise ValueError("ML-задача не найдена")
    if task.user_id != user_id:
        raise ValueError("Эта задача не пренадлежит текущему пользователю")

    return task


def get_model_by_code(session: Session, model_code: str) -> MLModel | None:
    return session.scalar(
        select(MLModel).where(MLModel.model_id == model_code)
    )


def submit_prediction(session: Session, user_id: int, model_code: str, data: str) -> MLTask:
    if not data or not data.strip():
        raise ValueError("Данные для предсказания не могут быть пустыми")

    user = session.get(User, user_id)
    if user is None:
        raise ValueError("Пользователь не найден")

    model = get_model_by_code(session, model_code)
    if model is None:
        raise ValueError("ML-модель не найдена")

    if user.balance < model.prediction_price:
        raise ValueError("Недостаточно средств на балансе")

    try:
        task = MLTask(
            user_id=user.id,
            model_id=model.id,
            data=data,
            status=TaskStatus.PROCESSING,
        )
        session.add(task)

        session.flush()

        transaction = Transaction(
            user_id=user.id,
            task_id=task.id,
            amount=model.prediction_price,
            transaction_type=TransactionType.DEBIT,
        )
        session.add(transaction)

        user.balance -= model.prediction_price

        task.result = f"demo_prediction_for:{data}"
        task.status = TaskStatus.COMPLETED

        session.commit()
        session.refresh(task)
        return task

    except Exception:
        session.rollback()
        raise


def enqueue_prediction(session: Session, user_id: int, model_code: str, data: str) -> MLTask:
    if not data or not data.strip():
        raise ValueError("Данные для предсказания не могут быть пустыми")

    user = session.get(User, user_id)
    if user is None:
        raise ValueError("Пользователь не найден")

    model = get_model_by_code(session, model_code)
    if model is None:
        raise ValueError("ML-модель не найдена")

    price = model.prediction_price
    if user.balance < price:
        raise ValueError("Недостаточно средств на балансе")

    try:
        task = MLTask(
            user_id=user.id,
            model_id=model.id,
            data=data,
            status=TaskStatus.CREATED,
        )
        session.add(task)
        session.flush()

        transaction = Transaction(
            user_id=user.id,
            task_id=task.id,
            amount=price,
            transaction_type=TransactionType.DEBIT,
        )
        session.add(transaction)

        user.balance -= price

        session.commit()
        session.refresh(task)

    except Exception:
        session.rollback()
        raise

    try:
        message = {
            "task_id": task.id,
            "user_id": user.id,
            "model": model.model_id,
            "data": data,
            "timestamp": datetime.now().isoformat(),
        }
        publish_prediction_task(message)
        return task

    except Exception as exc:
        try:
            failed_task = session.get(MLTask, task.id)
            failed_user = session.get(User, user.id)

            if failed_task is not None:
                failed_task.status = TaskStatus.FAILED
                failed_task.result = "Queue publish error"

            if failed_user is not None:
                failed_user.balance += price
                session.add(
                    Transaction(
                        user_id=failed_user.id,
                        task_id=task.id,
                        amount=price,
                        transaction_type=TransactionType.CREDIT,
                    )
                )

            session.commit()

        except Exception:
            session.rollback()

        raise RuntimeError(
            "Не удалось отправить задачу в очередь RabbitMQ") from exc


def get_available_models(session: Session) -> list[MLModel]:
    return list(
        session.scalars(
            select(MLModel).order_by(MLModel.model_id.asc())
        ).all()
    )
