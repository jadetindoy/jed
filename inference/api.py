"""
inference/api.py
----------------
FastAPI REST server exposing the JedAI model as an HTTP API.

Endpoints:
  POST /complete    — raw text completion
  POST /chat        — simple chat-style completion with a system prompt
  GET  /health      — liveness check

Launch:
    uvicorn inference.api:app --host 0.0.0.0 --port 8000

Or via Python:
    python -m inference.api --checkpoint checkpoints/ckpt_0010000.pt
"""

import argparse
import logging
from typing import Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

log = logging.getLogger(__name__)

app = FastAPI(
    title="JedAI Language Model API",
    description="Decoder-only Transformer text generation API",
    version="0.1.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global generator — loaded once at startup
_generator = None


# ---------------------------------------------------------------------------
# Request / Response schemas
# ---------------------------------------------------------------------------

class CompletionRequest(BaseModel):
    prompt: str = Field(..., description="Input text to complete")
    max_new_tokens: int = Field(256, ge=1, le=2048)
    temperature: float = Field(0.8, ge=0.0, le=2.0)
    top_k: int = Field(50, ge=0)
    top_p: float = Field(0.95, ge=0.0, le=1.0)


class CompletionResponse(BaseModel):
    prompt: str
    completion: str
    full_text: str


class ChatRequest(BaseModel):
    user_message: str
    system_prompt: str = "You are JedAI, a helpful assistant."
    max_new_tokens: int = Field(512, ge=1, le=2048)
    temperature: float = Field(0.7, ge=0.0, le=2.0)
    top_k: int = Field(50, ge=0)
    top_p: float = Field(0.95, ge=0.0, le=1.0)


class ChatResponse(BaseModel):
    user_message: str
    assistant_response: str


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.get("/health")
async def health() -> dict:
    return {"status": "ok", "model_loaded": _generator is not None}


@app.post("/complete", response_model=CompletionResponse)
async def complete(request: CompletionRequest) -> CompletionResponse:
    if _generator is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    full_text = _generator.generate(
        prompt=request.prompt,
        max_new_tokens=request.max_new_tokens,
        temperature=request.temperature,
        top_k=request.top_k,
        top_p=request.top_p,
    )
    completion = full_text[len(request.prompt):]
    return CompletionResponse(prompt=request.prompt, completion=completion, full_text=full_text)


@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest) -> ChatResponse:
    if _generator is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    prompt = (
        f"<|system|>\n{request.system_prompt}\n"
        f"<|user|>\n{request.user_message}\n"
        f"<|assistant|>\n"
    )
    full_text = _generator.generate(
        prompt=prompt,
        max_new_tokens=request.max_new_tokens,
        temperature=request.temperature,
        top_k=request.top_k,
        top_p=request.top_p,
    )
    return ChatResponse(user_message=request.user_message, assistant_response=full_text[len(prompt):])


# ---------------------------------------------------------------------------
# Startup / CLI
# ---------------------------------------------------------------------------

def load_model(checkpoint: str, tokenizer: str, device: Optional[str] = None) -> None:
    global _generator
    from inference.generate import Generator
    log.info(f"Loading model from {checkpoint} ...")
    _generator = Generator(model_path=checkpoint, tokenizer_path=tokenizer, device=device)
    log.info("Model ready.")


if __name__ == "__main__":
    import uvicorn
    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser(description="JedAI inference API server")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--tokenizer", default="tokenizer/tokenizer.json")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--device", default=None)
    args = parser.parse_args()
    load_model(args.checkpoint, args.tokenizer, args.device)
    uvicorn.run(app, host=args.host, port=args.port)
