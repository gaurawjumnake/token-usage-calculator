import os
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware

from backend.api.recommendation_api import router as recommendation_router
from backend.api.questionare_api import router as questionare_router
from backend.api.models_api import router as models_router
from backend.utilites.app_logger import Logger
log = Logger()

app = FastAPI(
    title="Token Usage Calculator API",
    description="API for calculating token usage and providing LLM recommendations based on user input.",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

ORIGIN_SECRET = os.getenv("ORIGIN_SECRET", "")

class OriginSecretMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        if ORIGIN_SECRET and request.url.path not in ("/health", "/"):
            if request.headers.get("X-Origin-Secret") != ORIGIN_SECRET:
                return JSONResponse(status_code=403, content={"detail": "Forbidden"})
        return await call_next(request)

app.add_middleware(OriginSecretMiddleware)

prefix="/api/v1"

app.include_router(recommendation_router, prefix = prefix , tags=["Recommendations"])
app.include_router(questionare_router, prefix = prefix, tags=["Questionnaire"])
app.include_router(models_router, prefix = prefix, tags=["Models"])
log.log_info("All routers are up...")

@app.get("/")
def read_root():
    return {"status":200,
        "message": "Welcome to the Token Usage Calculator API"}


@app.get("/health")
def health_check():
    return {"status":200}

from mangum import Mangum
handler = Mangum(app, lifespan="auto")



if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="localhost", port=8001, reload=True)

