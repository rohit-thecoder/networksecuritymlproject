import sys
import os
import certifi
import pandas as pd
import pymongo

from dotenv import load_dotenv
from fastapi import FastAPI, File, UploadFile, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response
from starlette.responses import RedirectResponse
from starlette.templating import Jinja2Templates
from uvicorn import run as app_run

from networksecurity.exception.exception import NetworkSecurityException
from networksecurity.pipeline.training_pipeline import TrainingPipeline
from networksecurity.utils.main_utils.utils import load_object
from networksecurity.utils.ml_utils.model.estimator import NetworkModel

# ===================== ENV SETUP =====================
ca = certifi.where()
basedir = os.path.abspath(os.path.dirname(__file__))
load_dotenv(os.path.join(basedir, '.env'))

mongo_db_url = os.getenv("MONGO_DB_URL")

# ===================== DB CONNECTION =====================
client = pymongo.MongoClient(mongo_db_url, tlsCAFile=ca)

from networksecurity.constant.training_pipeline import (
    DATA_INGESTION_COLLECTION_NAME,
    DATA_INGESTION_DATABASE_NAME
)

database = client[DATA_INGESTION_DATABASE_NAME]
collection = database[DATA_INGESTION_COLLECTION_NAME]

# ===================== FASTAPI APP =====================
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ===================== TEMPLATE SETUP =====================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

templates = Jinja2Templates(directory="templates")  # simple path (safe)

# ===================== ROUTES =====================

@app.get("/", tags=["authentication"])
async def index():
    return RedirectResponse(url="/docs")


@app.get("/train")
async def train_route():
    try:
        train_pipeline = TrainingPipeline()
        train_pipeline.run_pipeline()
        return Response("Training successful")
    except Exception as e:
        raise NetworkSecurityException(e, sys)


@app.post("/predict")
async def predict_route(request: Request, file: UploadFile = File(...)):
    try:
        # Load Data
        df = pd.read_csv(file.file)

        # Load Model
        preprocessor = load_object("final_model/preprocessor.pkl")
        model = load_object("final_model/model.pkl")

        network_model = NetworkModel(
            preprocessor=preprocessor,
            model=model
        )

        # Prediction
        y_pred = network_model.predict(df)
        df['predicted_column'] = y_pred

        # Save Output
        os.makedirs("prediction_output", exist_ok=True)
        df.to_csv("prediction_output/output.csv", index=False)

        # HTML Table
        table_html = df.to_html(classes="table table-striped")

        # FIX: Explicitly name the arguments to avoid the "unhashable dict" error
        return templates.TemplateResponse(
            request=request, 
            name="table.html", 
            context={"table": table_html}
        )

    except Exception as e:
        print("ERROR:", e)
        # Re-raising using your custom exception
        raise NetworkSecurityException(e, sys)


# ===================== MAIN =====================
if __name__ == "__main__":
    app_run(app, host="localhost", port=8000)