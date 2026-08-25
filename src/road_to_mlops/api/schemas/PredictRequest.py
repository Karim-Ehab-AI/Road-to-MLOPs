from pydantic import BaseModel, Field
from typing import Literal

class PredictRequest(BaseModel):
    sepal_length: float = Field(..., gt=0)
    sepal_width: float = Field(..., gt=0)
    petal_length: float = Field(..., gt=0)
    petal_width: float = Field(..., gt=0)

class ResponseSchema(BaseModel):
    label: Literal[0, 1, 2]
    confidence: float = Field(..., ge=0, le=1)
