"""
FastAPI backend for BlueBot. Wraps src.pipeline.ask() in a REST API.
Run from project root: uvicorn src.api:app --reload
"""

import time

from fastapi import FastAPI
from pydantic import BaseModel

from src.pipeline import ask

app = FastAPI(title="BlueBot API")


class ChatRequest(BaseModel):
    query: str


class ChatResponse(BaseModel):
    answer: str
    sources: list[str]
    is_relevant: bool
    response_time: float


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.post("/chat", response_model=ChatResponse)
def chat(request: ChatRequest) -> ChatResponse:
    started = time.perf_counter()
    result = ask(request.query)
    elapsed = time.perf_counter() - started
    return ChatResponse(
        answer=result["answer"],
        sources=result["sources"],
        is_relevant=result["is_relevant"],
        response_time=elapsed,
    )