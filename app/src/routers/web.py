from fastapi import APIRouter, Request, Form, Depends
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.templating import Jinja2Templates
from sqlalchemy.orm import Session
from services import get_user_by_id, deposit_balance
from services import get_user_tasks_history, get_user_transactions_history, get_available_models
from services import enqueue_prediction, get_task_by_id
from fastapi import UploadFile, File

from dependencies import get_db
from models import User
from services import create_user

router = APIRouter(tags=["web"])

templates = Jinja2Templates(directory="src/templates")


@router.get("/", response_class=HTMLResponse)
def home_page(request: Request):
    return templates.TemplateResponse(
        request=request,
        name="index.html",
        context={
            "request": request,
            "page_title": "ML Service",
        },
    )


@router.get("/register", response_class=HTMLResponse)
def register_page(request: Request):
    return templates.TemplateResponse(
        request=request,
        name="register.html",
        context={
            "request": request,
            "page_title": "Регистрация",
        },
    )


@router.post("/register", response_class=HTMLResponse)
def register_submit(
    request: Request,
    login: str = Form(...),
    password: str = Form(...),
    db: Session = Depends(get_db),
):
    try:
        create_user(db, login=login, password_hash=password)

        return templates.TemplateResponse(
            request=request,
            name="register.html",
            context={
                "request": request,
                "page_title": "Регистрация",
                "success": "Регистрация успешна. Теперь войдите в систему.",
            },
        )
    except ValueError as e:
        return templates.TemplateResponse(
            request=request,
            name="register.html",
            context={
                "request": request,
                "page_title": "Регистрация",
                "error": str(e),
            },
        )


@router.get("/login", response_class=HTMLResponse)
def login_page(request: Request):
    return templates.TemplateResponse(
        request=request,
        name="login.html",
        context={
            "request": request,
            "page_title": "Вход",
        },
    )


@router.post("/login", response_class=HTMLResponse)
def login_submit(
    request: Request,
    login: str = Form(...),
    password: str = Form(...),
    db: Session = Depends(get_db),
):
    user = db.query(User).filter(User.login == login).first()

    if not user or user.password_hash != password:
        return templates.TemplateResponse(
            request=request,
            name="login.html",
            context={
                "request": request,
                "page_title": "Вход",
                "error": "Неверный логин или пароль",
            },
        )

    request.session["user_id"] = user.id

    return RedirectResponse(url="/dashboard", status_code=303)


@router.get("/logout")
def logout(request: Request):
    request.session.clear()
    return RedirectResponse(url="/", status_code=303)


def get_current_web_user(request: Request, db: Session) -> User | None:
    user_id = request.session.get("user_id")
    if not user_id:
        return None
    return db.query(User).filter(User.id == user_id).first()


@router.get("/dashboard", response_class=HTMLResponse)
def dashboard_page(
    request: Request,
    db: Session = Depends(get_db),
):
    user = get_current_web_user(request, db)
    if not user:
        return RedirectResponse(url="/login", status_code=303)

    return templates.TemplateResponse(
        request=request,
        name="dashboard.html",
        context={
            "request": request,
            "page_title": "Личный кабинет",
            "user": user,
        },
    )


@router.post("/dashboard/deposit", response_class=HTMLResponse)
def dashboard_deposit(
    request: Request,
    amount: int = Form(...),
    db: Session = Depends(get_db),
):
    user = get_current_web_user(request, db)
    if not user:
        return RedirectResponse(url="/login", status_code=303)

    try:
        deposit_balance(db, user.id, amount)
        user = get_user_by_id(db, user.id)

        return templates.TemplateResponse(
            request=request,
            name="dashboard.html",
            context={
                "request": request,
                "page_title": "Личный кабинет",
                "user": user,
                "success": "Баланс успешно пополнен",
            },
        )
    except ValueError as e:
        return templates.TemplateResponse(
            request=request,
            name="dashboard.html",
            context={
                "request": request,
                "page_title": "Личный кабинет",
                "user": user,
                "error": str(e),
            },
        )


@router.get("/history", response_class=HTMLResponse)
def history_page(
    request: Request,
    db: Session = Depends(get_db),
):
    user = get_current_web_user(request, db)
    if not user:
        return RedirectResponse(url="/login", status_code=303)

    tasks = get_user_tasks_history(db, user.id)
    transactions = get_user_transactions_history(db, user.id)

    return templates.TemplateResponse(
        request=request,
        name="history.html",
        context={
            "request": request,
            "page_title": "История",
            "user": user,
            "tasks": tasks,
            "transactions": transactions,
        },
    )


@router.get("/predict", response_class=HTMLResponse)
def predict_page(
    request: Request,
    db: Session = Depends(get_db),
):
    user = get_current_web_user(request, db)
    if not user:
        return RedirectResponse(url="/login", status_code=303)

    models = get_available_models(db)

    return templates.TemplateResponse(
        request=request,
        name="predict.html",
        context={
            "request": request,
            "page_title": "Предсказание",
            "user": user,
            "models": models,
        },
    )


@router.post("/predict-ui", response_class=HTMLResponse)
def predict_submit(
    request: Request,
    model_code: str = Form(...),
    file: UploadFile = File(...),
    db: Session = Depends(get_db),
):
    user = get_current_web_user(request, db)
    if not user:
        return RedirectResponse(url="/login", status_code=303)

    try:
        result = enqueue_prediction(db, user.id, model_code, file.filename)

        return templates.TemplateResponse(
            request=request,
            name="predict.html",
            context={
                "request": request,
                "page_title": "Предсказание",
                "user": get_user_by_id(db, user.id),
                "success": "Задача успешно отправлена",
                "task_id": result.id,
                "task_status": result.status,
                "task_message": "Task accepted and sent to RabbitMQ queue",
            },
        )
    except ValueError as e:
        return templates.TemplateResponse(
            request=request,
            name="predict.html",
            context={
                "request": request,
                "page_title": "Предсказание",
                "user": user,
                "error": str(e),
            },
        )
    except RuntimeError as e:
        return templates.TemplateResponse(
            request=request,
            name="predict.html",
            context={
                "request": request,
                "page_title": "Предсказание",
                "user": user,
                "error": str(e),
            },
        )


@router.get("/predict/{task_id}", response_class=HTMLResponse)
def predict_status_page(
    task_id: int,
    request: Request,
    db: Session = Depends(get_db),
):
    user = get_current_web_user(request, db)
    if not user:
        return RedirectResponse(url="/login", status_code=303)

    try:
        task = get_task_by_id(db, task_id, user.id)

        return templates.TemplateResponse(
            request=request,
            name="predict.html",
            context={
                "request": request,
                "page_title": "Предсказание",
                "user": user,
                "task_id": task.id,
                "task_status": task.status,
                "task_result": task.result,
                "task_created_at": task.created_at,
            },
        )
    except ValueError as e:
        return templates.TemplateResponse(
            request=request,
            name="predict.html",
            context={
                "request": request,
                "page_title": "Предсказание",
                "user": user,
                "error": str(e),
            },
        )
