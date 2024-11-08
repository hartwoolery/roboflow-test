from fastapi import APIRouter, UploadFile, File, Form
from app.models.florence_wrapper import FlorenceModelWrapper
from PIL import Image
import io
from typing import Optional

router = APIRouter()
model_wrapper = FlorenceModelWrapper()

@router.post("/predict/image")
async def predict_image(
    file: UploadFile = File(...),
    task: Optional[str] = Form(None),        # Required parameter for task type
    text: Optional[str] = Form(None)  # Optional text parameter
):
    image = Image.open(io.BytesIO(await file.read()))
    prediction = model_wrapper.predict_image(image, task, text)
    return prediction