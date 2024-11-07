from fastapi import APIRouter, UploadFile, File
from app.models.florence_wrapper import FlorenceModelWrapper
from PIL import Image
import io

router = APIRouter()
model_wrapper = FlorenceModelWrapper()

@router.post("/predict/image")
async def predict_image(file: UploadFile = File(...)):
    image = Image.open(io.BytesIO(await file.read()))
    prediction = model_wrapper.predict_image(image)
    return prediction