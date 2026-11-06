
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from backend.api.routes import router

app = FastAPI(
    title="DocuMind API",
    description="Intelligent PDF Question Answering System using RAG",
    version="1.0.0"
)

# Allow the local frontend to communicate with the backend
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://127.0.0.1:5500",
        "http://localhost:5500"
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)

# Register API routes
app.include_router(router)


@app.get("/")
async def root():
    return {
        "message": "Welcome to DocuMind API"
    }
