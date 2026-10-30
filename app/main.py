"""FastAPI application entry point."""

from fastapi import FastAPI

from app.api.routes import router


app = FastAPI(
    title="LabInsight API",
    version="0.1.0",
    description=(
        "Experimental statistical analysis of public NHANES laboratory data. "
        "It does not provide diagnoses or treatment recommendations."
    ),
)
app.include_router(router)
