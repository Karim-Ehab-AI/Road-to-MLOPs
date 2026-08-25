from contextlib import asynccontextmanager
from pathlib import Path

import onnxruntime as ort
from fastapi import FastAPI

from .api.routes import base_router, prdict_router


MODEL_PATH = Path(__file__).resolve().parents[2] / "artifacts" / "model.onnx"


@asynccontextmanager
async def lifespan(app: FastAPI):
	app.session = ort.InferenceSession(str(MODEL_PATH))
	app.input_name = app.session.get_inputs()[0].name
	yield


app = FastAPI(lifespan=lifespan)
app.include_router(base_router)
app.include_router(prdict_router)


def main() -> None:
	import uvicorn

	uvicorn.run("road_to_mlops.app:app", host="0.0.0.0", port=8000)

