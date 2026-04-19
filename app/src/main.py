from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from starlette.middleware.sessions import SessionMiddleware

from routers.auth import router as auth_router
from routers.users import router as user_router
from routers.balance import router as balance_router
from routers.history import router as history_router
from routers.predict import router as predict_router
from routers.web import router as web_router


app = FastAPI(
    title="Image Classification ML Service",
    description="REST API for image classification ML service",
    version="1.0.0",
)
app.mount("/static", StaticFiles(directory="src/static"), name="static")
app.add_middleware(SessionMiddleware, secret_key="super-secret-key")


@app.get("/health")
def health():
    return {"status": "ok"}


app.include_router(auth_router)
app.include_router(user_router)
app.include_router(balance_router)
app.include_router(history_router)
app.include_router(predict_router)
app.include_router(web_router)
