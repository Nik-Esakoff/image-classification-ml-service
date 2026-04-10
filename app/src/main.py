from fastapi import FastAPI
from routers.auth import router as auth_router
from routers.users import router as user_router
from routers.balance import router as balance_router
from routers.history import router as history_router
from routers.predict import router as predict_router


app = FastAPI(
    title="Image Classification ML Service",
    description="REST API for image classification ML service",
    version="1.0.0",
)


@app.get("/health")
def health():
    return {"status": "ok"}


app.include_router(auth_router)
app.include_router(user_router)
app.include_router(balance_router)
app.include_router(history_router)
app.include_router(predict_router)
