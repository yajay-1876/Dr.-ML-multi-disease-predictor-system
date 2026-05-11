from dotenv import load_dotenv
load_dotenv()

from fastapi import FastAPI

from src.backend.api.routes import router

app=FastAPI(
    title="Multi-disease-predictor",
    version="1.0.0",
    description="For the given input from diagnosis , This ML predictor can predict severity of diabetes and heart disease"
)

app.include_router(router=router,prefix="/api")
