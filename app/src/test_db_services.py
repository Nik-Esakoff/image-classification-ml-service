import time

from db import SessionLocal
from models import MLModel, UserRole
from services import (
    create_task,
    create_user,
    deposit_balance,
    get_user_by_id,
    get_user_by_login,
    get_user_tasks_history,
    get_user_transactions_history,
    withdraw_balance,
)


def main():
    session_local = SessionLocal()
    try:
        print('создание пользователя:')
        new_login = f"user_log_test_{int(time.time())}"
        print(new_login)
        test_user = create_user(session_local, new_login, '1234', role=UserRole.USER, balance=100)
        print(test_user.id, test_user.login, test_user.role, test_user.password_hash)

        print('проверяем пользователя в базе по логину:')
        user_by_login = get_user_by_login(session_local, new_login)
        print(f'найденный по логину пользователь: id: {user_by_login.id}, ', 
            f'balance: {user_by_login.balance}, password: {user_by_login.password_hash}')
        
        print('проверяем пользователя в базе по id')
        user_by_id = get_user_by_id(session_local, test_user.id)
        print(f'найденный по id пользователь: login: {user_by_id.login}, ', 
            f'balance: {user_by_id.balance}, password: {user_by_id.password_hash}')

        print("создаем транзакцию пополнения баланса")
        dep_t = deposit_balance(session_local, test_user.id, 100)
        print(f'транзакция: amount: {dep_t.amount}, created_at: {dep_t.created_at}, task: {dep_t.task_id}, type: {dep_t.transaction_type}')

        model = session_local.query(MLModel).filter_by(model_id="resnet18").first()
        if model is None:
            raise ValueError("Не найдена демо-модель resnet18")

        print("создаем ml задачу")
        task_test = create_task(session_local, test_user.id, model.id, "test_data.png")
        print(f"task: created_at: {task_test.created_at}, data: {task_test.data}, status: {task_test.status}")

        print("создаем транзакцию списания средств")
        credit_t = withdraw_balance(session_local, test_user.id, 50, task_test.id)
        print(f'транзакция: amount: {credit_t.amount}, created_at: {credit_t.created_at}, task: {credit_t.task_id}, type: {credit_t.transaction_type}')

        user_after_transactions = get_user_by_id(session_local, test_user.id)
        print("проверяем пользователя после транзакций")
        print(f"user: login: {user_after_transactions.login}, balance: {user_after_transactions.balance}, tr_his: {user_after_transactions.transactions}")
        print(f"tasks_his: {user_after_transactions.tasks}")


        print("проверяем историю транзакций пользователя:")
        transactions = get_user_transactions_history(session_local, test_user.id)
        for credit_t in transactions:
            print(
                f"  tx_id: {credit_t.id}, type: {credit_t.transaction_type}, amount: {credit_t.amount}, task_id: {credit_t.task_id}, created_at: {credit_t.created_at}"
            )

        print("проверяем историю задач пользователя:")
        tasks = get_user_tasks_history(session_local, test_user.id)
        for task_test in tasks:
            print(
                f"  task_id: {task_test.id}, model_id: {task_test.model_id}, status: {task_test.status}, created_at: {task_test.created_at}"
            )


        print("проверяем ошибку на  нехватку средств:")
        try:
            withdraw_balance(session_local, test_user.id, 50000, None)
        except ValueError as e:
                print(f"ожидаемая ошибка: {e}")
    finally:
        session_local.close()



if __name__ == "__main__":
    main()