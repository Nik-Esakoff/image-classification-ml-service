import requests
import uuid
import time
from requests.auth import HTTPBasicAuth

URL = "http://localhost:8000"
MODEL_ID = "resnet18"
MODEL_PRICE = 10


def test_healthcheck():
    answer = requests.get(f"{URL}/health")
    assert answer.status_code == 200
    assert answer.json() == {"status": "ok"}


def test_user_registation():
    login, password = f"test_user_{uuid.uuid4().hex[:4]}", 'test_password'
    answer = requests.post(f"{URL}/auth/register",
                           json={"login": login, "password": password})
    assert answer.status_code == 201

    reg_user = answer.json()
    assert reg_user["login"] == login
    assert reg_user["balance"] == 100


def test_user_authorization():
    login, password = f"test_user_{uuid.uuid4().hex[:4]}", 'test_password'

    register_answer = requests.post(f"{URL}/auth/register",
                                    json={"login": login, "password": password})
    assert register_answer.status_code == 201

    answer = requests.get(f"{URL}/users/me",
                          auth=HTTPBasicAuth(login, password))
    assert answer.status_code == 200

    auth_user = answer.json()
    assert auth_user["login"] == login


def test_user_authorization_wrong_password():
    login, password = f"test_user_{uuid.uuid4().hex[:4]}", 'test_password'

    register_answer = requests.post(f"{URL}/auth/register",
                                    json={"login": login, "password": password})
    assert register_answer.status_code == 201

    answer = requests.get(f"{URL}/users/me",
                          auth=HTTPBasicAuth(login, "111"))
    assert answer.status_code == 401


def test_duplicate_registration():
    login, password = f"test_user_{uuid.uuid4().hex[:4]}", 'test_password'

    register_answer = requests.post(f"{URL}/auth/register",
                                    json={"login": login, "password": password})
    assert register_answer.status_code == 201

    register_answer_2 = requests.post(f"{URL}/auth/register",
                                      json={"login": login, "password": password})
    assert register_answer_2.status_code == 409


def test_initial_balance():
    login, password = f"test_user_{uuid.uuid4().hex[:4]}", 'test_password'
    answer = requests.post(f"{URL}/auth/register",
                           json={"login": login, "password": password})
    assert answer.status_code == 201

    balance = requests.get(f"{URL}/balance/",
                           auth=HTTPBasicAuth(login, password))
    assert balance.status_code == 200

    balance_json = balance.json()
    assert balance_json["balance"] == 100


def test_balance_deposit():
    login, password = f"test_user_{uuid.uuid4().hex[:4]}", 'test_password'
    answer = requests.post(f"{URL}/auth/register",
                           json={"login": login, "password": password})
    assert answer.status_code == 201

    balance_dep = requests.post(f"{URL}/balance/deposit",
                                auth=HTTPBasicAuth(login, password),
                                json={"amount": 100})
    assert balance_dep.status_code == 200
    assert balance_dep.json()["balance"] == 200


def test_successful_prediction_returns_202():
    login, password = f"test_user_{uuid.uuid4().hex[:4]}", 'test_password'
    answer = requests.post(f"{URL}/auth/register",
                           json={"login": login, "password": password})
    assert answer.status_code == 201

    predict = requests.post(f"{URL}/predict",
                            auth=HTTPBasicAuth(login, password),
                            json={"model_id": MODEL_ID, "data": "cat.png"})
    assert predict.status_code == 202
    assert predict.json()["task_id"] is not None


def test_successful_prediction_debits_balance():
    login, password = f"test_user_{uuid.uuid4().hex[:4]}", 'test_password'
    answer = requests.post(f"{URL}/auth/register",
                           json={"login": login, "password": password})
    assert answer.status_code == 201

    old_balance = requests.get(f"{URL}/balance/",
                               auth=HTTPBasicAuth(login, password))
    assert old_balance.status_code == 200

    old_balance_json = old_balance.json()
    assert old_balance_json["balance"] == 100

    predict = requests.post(f"{URL}/predict",
                            auth=HTTPBasicAuth(login, password),
                            json={"model_id": MODEL_ID, "data": "cat.png"})
    assert predict.status_code == 202
    new_balance = requests.get(f"{URL}/balance/",
                               auth=HTTPBasicAuth(login, password))
    assert new_balance.status_code == 200

    balance_json = new_balance.json()
    assert balance_json["balance"] == old_balance_json["balance"] - 10


def test_unknown_model_returns_404():
    login, password = f"test_user_{uuid.uuid4().hex[:4]}", 'test_password'
    answer = requests.post(f"{URL}/auth/register",
                           json={"login": login, "password": password})
    assert answer.status_code == 201

    old_balance = requests.get(f"{URL}/balance/",
                               auth=HTTPBasicAuth(login, password))
    assert old_balance.status_code == 200

    old_balance_json = old_balance.json()
    assert old_balance_json["balance"] == 100

    predict = requests.post(f"{URL}/predict",
                            auth=HTTPBasicAuth(login, password),
                            json={"model_id": 'unknown', "data": "cat.png"},
                            timeout=10)
    assert predict.status_code == 404

    new_balance = requests.get(f"{URL}/balance/",
                               auth=HTTPBasicAuth(login, password))
    assert new_balance.status_code == 200

    balance_json = new_balance.json()
    assert balance_json["balance"] == old_balance_json["balance"]


def test_invalid_prediction_data_returns_422():
    login, password = f"test_user_{uuid.uuid4().hex[:4]}", 'test_password'
    answer = requests.post(f"{URL}/auth/register",
                           json={"login": login, "password": password})
    assert answer.status_code == 201

    old_balance = requests.get(f"{URL}/balance/",
                               auth=HTTPBasicAuth(login, password))
    assert old_balance.status_code == 200

    old_balance_json = old_balance.json()
    assert old_balance_json["balance"] == 100

    predict = requests.post(f"{URL}/predict",
                            auth=HTTPBasicAuth(login, password),
                            json={"model_id": MODEL_ID, "data": 111},
                            timeout=10)
    assert predict.status_code == 422

    new_balance = requests.get(f"{URL}/balance/",
                               auth=HTTPBasicAuth(login, password))
    assert new_balance.status_code == 200

    balance_json = new_balance.json()
    assert balance_json["balance"] == old_balance_json["balance"]


def test_invalid_prediction_data_returns_400():
    login, password = f"test_user_{uuid.uuid4().hex[:4]}", 'test_password'
    answer = requests.post(f"{URL}/auth/register",
                           json={"login": login, "password": password})
    assert answer.status_code == 201

    old_balance = requests.get(f"{URL}/balance/",
                               auth=HTTPBasicAuth(login, password))
    assert old_balance.status_code == 200

    old_balance_json = old_balance.json()
    assert old_balance_json["balance"] == 100

    predict = requests.post(f"{URL}/predict",
                            auth=HTTPBasicAuth(login, password),
                            json={"model_id": MODEL_ID, "data": "   "},
                            timeout=10)
    assert predict.status_code == 400

    new_balance = requests.get(f"{URL}/balance/",
                               auth=HTTPBasicAuth(login, password))
    assert new_balance.status_code == 200

    balance_json = new_balance.json()
    assert balance_json["balance"] == old_balance_json["balance"]


def test_insufficient_balance_returns_409():
    login, password = f"test_user_{uuid.uuid4().hex[:4]}", 'test_password'
    answer = requests.post(
        f"{URL}/auth/register",
        json={"login": login, "password": password},
        timeout=10
    )
    assert answer.status_code == 201

    for i in range(10):
        predict = requests.post(
            f"{URL}/predict",
            auth=HTTPBasicAuth(login, password),
            json={"model_id": MODEL_ID, "data": f"cat_{i}.png"},
            timeout=10
        )
        assert predict.status_code == 202

    balance = requests.get(
        f"{URL}/balance/",
        auth=HTTPBasicAuth(login, password),
        timeout=10
    )
    assert balance.status_code == 200
    assert balance.json()["balance"] == 0

    extra_predict = requests.post(
        f"{URL}/predict",
        auth=HTTPBasicAuth(login, password),
        json={"model_id": MODEL_ID, "data": "one_more_cat.png"},
        timeout=10
    )
    assert extra_predict.status_code == 409

    final_balance = requests.get(
        f"{URL}/balance/",
        auth=HTTPBasicAuth(login, password),
        timeout=10
    )
    assert final_balance.status_code == 200
    assert final_balance.json()["balance"] == 0


def test_successful_prediction_eventually_returns_result():
    login, password = f"test_user_{uuid.uuid4().hex[:4]}", 'test_password'
    answer = requests.post(
        f"{URL}/auth/register",
        json={"login": login, "password": password},
        timeout=10
    )
    assert answer.status_code == 201

    predict = requests.post(
        f"{URL}/predict",
        auth=HTTPBasicAuth(login, password),
        json={"model_id": MODEL_ID, "data": "bird.png"},
        timeout=10
    )
    assert predict.status_code == 202

    task_id = predict.json()["task_id"]

    final_task = None
    for _ in range(30):
        task = requests.get(
            f"{URL}/history/tasks/{task_id}",
            auth=HTTPBasicAuth(login, password),
            timeout=10
        )
        assert task.status_code == 200

        final_task = task.json()
        if str(final_task["status"]).lower() in ["completed", "failed"]:
            break

        time.sleep(1)

    assert final_task is not None
    assert str(final_task["status"]).lower() == "completed"
    assert final_task["result"] is not None
    assert final_task["result"].startswith("demo_prediction_for:")


def test_transactions_history_contains_credit_and_debit():
    login, password = f"test_user_{uuid.uuid4().hex[:4]}", 'test_password'
    answer = requests.post(
        f"{URL}/auth/register",
        json={"login": login, "password": password},
        timeout=10
    )
    assert answer.status_code == 201

    deposit = requests.post(
        f"{URL}/balance/deposit",
        auth=HTTPBasicAuth(login, password),
        json={"amount": 25},
        timeout=10
    )
    assert deposit.status_code == 200

    predict = requests.post(
        f"{URL}/predict",
        auth=HTTPBasicAuth(login, password),
        json={"model_id": MODEL_ID, "data": "history_cat.png"},
        timeout=10
    )
    assert predict.status_code == 202

    task_id = predict.json()["task_id"]

    history = requests.get(
        f"{URL}/history/transactions",
        auth=HTTPBasicAuth(login, password),
        timeout=10
    )
    assert history.status_code == 200

    transactions = history.json()

    assert any(
        str(tx["transaction_type"]).lower() == "credit" and tx["amount"] == 25
        for tx in transactions
    )

    assert any(
        str(tx["transaction_type"]).lower() == "debit"
        and tx["amount"] == MODEL_PRICE
        and tx["task_id"] == task_id
        for tx in transactions
    )


def test_tasks_history_contains_created_task():
    login, password = f"test_user_{uuid.uuid4().hex[:4]}", 'test_password'
    answer = requests.post(
        f"{URL}/auth/register",
        json={"login": login, "password": password},
        timeout=10
    )
    assert answer.status_code == 201

    predict = requests.post(
        f"{URL}/predict",
        auth=HTTPBasicAuth(login, password),
        json={"model_id": MODEL_ID, "data": "task_history_cat.png"},
        timeout=10
    )
    assert predict.status_code == 202

    task_id = predict.json()["task_id"]

    history = requests.get(
        f"{URL}/history/tasks",
        auth=HTTPBasicAuth(login, password),
        timeout=10
    )
    assert history.status_code == 200

    tasks = history.json()

    assert any(task["id"] == task_id for task in tasks)


def test_get_task_by_id_returns_expected_task():
    login, password = f"test_user_{uuid.uuid4().hex[:4]}", 'test_password'
    answer = requests.post(
        f"{URL}/auth/register",
        json={"login": login, "password": password},
        timeout=10
    )
    assert answer.status_code == 201

    predict = requests.post(
        f"{URL}/predict",
        auth=HTTPBasicAuth(login, password),
        json={"model_id": MODEL_ID, "data": "single_task_cat.png"},
        timeout=10
    )
    assert predict.status_code == 202

    task_id = predict.json()["task_id"]

    task = requests.get(
        f"{URL}/history/tasks/{task_id}",
        auth=HTTPBasicAuth(login, password),
        timeout=10
    )
    assert task.status_code == 200

    task_json = task.json()
    assert task_json["id"] == task_id
    assert task_json["data"] == "single_task_cat.png"
