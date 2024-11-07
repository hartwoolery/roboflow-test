from fastapi import FastAPI
from app.routers import predictions

app = FastAPI()

# Include prediction routes
app.include_router(predictions.router)