from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse
from .schemas import PredictRequest, ResponseSchema
import numpy as np

base_router = APIRouter()

@base_router.get("/health")
async def welcome():
    return JSONResponse(
        content={
            "signal": "Server is working :)"
        }
    )

prdict_router = APIRouter()

@prdict_router.post("/predict")
async def model_predict(
    predict_req: PredictRequest, request: Request
) -> ResponseSchema:
    features = np.array([[
        predict_req.sepal_length,
        predict_req.sepal_width,
        predict_req.petal_length,
        predict_req.petal_width,
    ]], dtype=np.float32)

    labels, probabilities = request.app.session.run(
        None,
        {request.app.input_name: features},
    )

    predicted_label = int(labels[0])
    confidence = float(probabilities[0][predicted_label])

    return ResponseSchema(
        label=predicted_label,
        confidence=confidence,
    )