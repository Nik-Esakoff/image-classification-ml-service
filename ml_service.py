import datetime
from PIL import Image
from enum import Enum

import torchvision.transforms as transforms


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


class User:
    """Пользователь ML-сервиса с балансом, ролью и историей задач."""

    def __init__(self, user_id, login, role: UserRole, password_hash):
        self.id = user_id
        self.password_hash = password_hash
        self.login = login
        self.role = role
        self.balance = 100
        self.tasks = []

    def get_balance(self):
        """Возвращает текущий баланс пользователя."""
        return self.balance

    def deposit(self, contribution: int):
        """Пополняет баланс пользователя на указанную сумму."""
        user_transaction = CreditTransaction(contribution)
        user_transaction.apply(self)

    def withdraw(self, contribution: int, ml_task):
        """Списывает средства с баланса пользователя за выполнение задачи."""
        user_transaction = DebitTransaction(contribution, ml_task)
        user_transaction.apply(self)

    def add_task(self, ml_task: "MLTask"):
        """Добавляет задачу в историю запросов пользователя."""
        self.tasks.append(ml_task)

    def get_tasks_history(self):
        """Возвращает историю задач пользователя."""
        return self.tasks


class MLTask:
    """Задача пользователя на выполнение предсказания ML-моделью."""

    def __init__(self, user: User, image, model: "MLModel"):
        self.user = user
        self.data = image
        self.status = TaskStatus.CREATED
        self.created_at = datetime.datetime.now()
        self.model = model
        self.result = None
        self.user.add_task(self)


class PredictionResult:
    """Результат предсказания, связанный с конкретной ML-задачей."""

    def __init__(self, ml_task, result):
        self.result = result
        self.task = ml_task


class Transaction:
    """Базовая транзакция изменения баланса пользователя."""

    def __init__(self, amount: float, ml_task=None):
        if amount <= 0:
            raise ValueError("amount должен быть > 0")
        self.amount = amount
        self.ml_task = ml_task
        self.created_at = datetime.datetime.now()

    def apply(self, user: User):
        raise NotImplementedError


class DebitTransaction(Transaction):
    """Транзакция списания средств с баланса пользователя."""

    def apply(self, user: User):
        """Уменьшает баланс пользователя при достаточном количестве средств."""
        self.user = user
        if user.balance >= self.amount:
            user.balance -= self.amount
        else:
            if self.ml_task is not None:
                self.ml_task.status = TaskStatus.FAILED
            raise ValueError("Недостаточно средств на балансе")


class CreditTransaction(Transaction):
    """Транзакция пополнения баланса пользователя."""

    def apply(self, user: User):
        """Увеличивает баланс пользователя на указанную сумму."""
        self.user = user
        user.balance += self.amount


class MLModel:
    """Доменная сущность ML-модели с описанием и стоимостью предсказания."""

    def __init__(self, model_id, prediction_price, description: str):
        self.model_id = model_id
        self.description = description
        self.prediction_price = prediction_price


class PredictionService:
    """Сервис запуска ML-модели и обработки пользовательских задач."""

    def __init__(self, ml_model: MLModel, model_weights):
        self.model_info = ml_model
        self.loaded_model = model_weights

    def data_to_tensor(self, data):
        """Преобразует входное изображение в тензор для модели."""
        if data:
            image = Image.open(data)
            transform = transforms.Compose([transforms.PILToTensor()])
            prepared_data = transform(image)
            return prepared_data

    def start_processing(self, ml_task: MLTask):
        """Запускает обработку задачи, списывает средства и возвращает результат."""
        ml_task.user.withdraw(self.model_info.prediction_price, ml_task)

        data = ml_task.data
        ml_task.status = TaskStatus.PROCESSING
        tensor_image = self.data_to_tensor(data)

        try:
            prediction = self.loaded_model.predict(tensor_image)
        except Exception:
            prediction = None

        if prediction is not None:
            ml_task.status = TaskStatus.COMPLETED
            response = PredictionResult(ml_task, prediction)
            ml_task.result = response
            return response
        else:
            ml_task.status = TaskStatus.FAILED
            return None