import json
import os

import pika

from db import SessionLocal
from models import MLModel, MLTask, TaskStatus, Transaction, TransactionType, User

RABBITMQ_HOST = os.getenv("RABBITMQ_HOST", "rabbitmq")
RABBITMQ_PORT = int(os.getenv("RABBITMQ_PORT", "5672"))
RABBITMQ_USER = os.getenv("RABBITMQ_USER", "ml_user")
RABBITMQ_PASSWORD = os.getenv("RABBITMQ_PASSWORD", "ml_password")
RABBITMQ_QUEUE = os.getenv("RABBITMQ_QUEUE", "ml_tasks")


def make_demo_prediction(data: str) -> str:
    if not data or not data.strip():
        raise ValueError("Пустые данные для предсказания")

    return f"demo_prediction_for:{data}"


def process_task(ch, method, properties, body):
    session = SessionLocal()
    task_id = None

    try:
        payload = json.loads(body)
        task_id = payload.get("task_id")
        data = payload.get("data")

        if task_id is None:
            raise ValueError("В сообщении нет task_id")

        task = session.get(MLTask, task_id)
        if task is None:
            raise ValueError(f"MLTask с id={task_id} не найдена")

        model = session.get(MLModel, task.model_id)
        if model is None:
            raise ValueError("ML-модель не найдена")

        price = model.prediction_price

        task.status = TaskStatus.PROCESSING
        session.commit()

        prediction_result = make_demo_prediction(data)

        task.result = prediction_result
        task.status = TaskStatus.COMPLETED
        session.commit()

        ch.basic_ack(delivery_tag=method.delivery_tag)
        print(f"[worker] task {task_id} completed")

    except Exception as exc:
        session.rollback()

        try:
            if task_id is not None:

                failed_task = session.get(MLTask, task_id)
                if failed_task is not None and failed_task.status != TaskStatus.COMPLETED:

                    failed_task.status = TaskStatus.FAILED
                    failed_task.result = f"Worker error: {exc}"

                    if price is None:
                        failed_model = session.get(
                            MLModel, failed_task.model_id)
                        if failed_model is not None:
                            price = failed_model.prediction_price

                    if price is not None:
                        failed_user = session.get(User, task.user_id)

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

        ch.basic_ack(delivery_tag=method.delivery_tag)
        print(f"[worker] task {task_id} failed: {exc}")

    finally:
        session.close()


def main():
    credentials = pika.PlainCredentials(RABBITMQ_USER, RABBITMQ_PASSWORD)
    parameters = pika.ConnectionParameters(
        host=RABBITMQ_HOST,
        port=RABBITMQ_PORT,
        credentials=credentials,
    )

    connection = pika.BlockingConnection(parameters)
    channel = connection.channel()

    channel.queue_declare(
        queue=RABBITMQ_QUEUE,
        durable=True,
    )

    channel.basic_qos(prefetch_count=1)

    channel.basic_consume(
        queue=RABBITMQ_QUEUE,
        on_message_callback=process_task,
        auto_ack=False,
    )

    print("[worker] waiting for messages...")
    channel.start_consuming()


if __name__ == "__main__":
    main()
