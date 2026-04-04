from sqlalchemy import select

from db import Base, SessionLocal, engine
from models import MLModel, User, UserRole


def create_tables():
    """Создаёт таблицы в БД на основе ORM-моделей."""
    Base.metadata.create_all(bind=engine)


def seed_users(session):
    """Добавляет демо-пользователей, если их ещё нет."""
    demo_user = session.scalar(
        select(User).where(User.login == "demo_user")
    )
    if demo_user is None:
        session.add(
            User(
                login="demo_user",
                password_hash="demo_user_hash",
                role=UserRole.USER,
                balance=100,
            )
        )

    demo_admin = session.scalar(
        select(User).where(User.login == "demo_admin")
    )
    if demo_admin is None:
        session.add(
            User(
                login="demo_admin",
                password_hash="demo_admin_hash",
                role=UserRole.ADMIN,
                balance=100,
            )
        )


def seed_models(session):
    """Добавляет базовые ML-модели, если их ещё нет."""
    base_models = [
        {
            "model_id": "resnet18",
            "description": "Demo ResNet18 image classifier",
            "prediction_price": 10,
        },
        {
            "model_id": "mobilenet_v2",
            "description": "Demo MobileNetV2 image classifier",
            "prediction_price": 8,
        },
    ]

    for model_data in base_models:
        existing_model = session.scalar(
            select(MLModel).where(MLModel.model_id == model_data["model_id"])
        )
        if existing_model is None:
            session.add(MLModel(**model_data))


def init_db():
    """Создаёт таблицы и заполняет БД начальными данными."""
    create_tables()

    session = SessionLocal()
    try:
        seed_users(session)
        seed_models(session)
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()


if __name__ == "__main__":
    init_db()