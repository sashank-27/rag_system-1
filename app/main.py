"""
FastAPI application entry point.

- Configures structured logging
- Initialises services on startup
- Registers all route routers
"""

from __future__ import annotations

from contextlib import asynccontextmanager
from pathlib import Path

import structlog
import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from app.config import settings
from app.routes import documents, query, servicenow_routes
from app.services.retrieval import retrieval_service
from app.services.servicenow import servicenow_client

# ── Structured logging setup ───────────────────────────────────────────────

structlog.configure(
    processors=[
        structlog.contextvars.merge_contextvars,
        structlog.processors.add_log_level,
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.dev.ConsoleRenderer(),
    ],
    wrapper_class=structlog.make_filtering_bound_logger(0),
    context_class=dict,
    logger_factory=structlog.PrintLoggerFactory(),
)

logger = structlog.get_logger(__name__)


# ── Lifespan (startup / shutdown) ──────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialise heavy services at startup; clean up on shutdown."""
    logger.info("starting_up")

    # Vector store
    retrieval_service.init()

    # ServiceNow client config
    servicenow_client.configure()

    logger.info("startup_complete", host=settings.host, port=settings.port)
    yield

    # Shutdown
    retrieval_service.close()
    logger.info("shutdown_complete")


# ── App factory ─────────────────────────────────────────────────────────────

app = FastAPI(
    title="Multilingual RAG System",
    description=(
        "Production-grade multilingual Retrieval-Augmented Generation system "
        "with ServiceNow CMDB integration."
    ),
    version="1.0.0",
    lifespan=lifespan,
)

# ── CORS ────────────────────────────────────────────────────────────────────

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Register routers ───────────────────────────────────────────────────────

app.include_router(documents.router)
app.include_router(query.router)
app.include_router(servicenow_routes.router)

# ── Static files & UI ──────────────────────────────────────────────────────

_STATIC_DIR = Path(__file__).parent / "static"
app.mount("/static", StaticFiles(directory=str(_STATIC_DIR)), name="static")


@app.get("/", include_in_schema=False)
async def serve_ui():
    """Serve the web UI."""
    return FileResponse(str(_STATIC_DIR / "index.html"))


# ── Health check ────────────────────────────────────────────────────────────

@app.get("/health", tags=["System"])
async def health():
    return {"status": "healthy"}


# ── CLI entry point ─────────────────────────────────────────────────────────

if __name__ == "__main__":
    uvicorn.run(
        "app.main:app",
        host=settings.host,
        port=settings.port,
        reload=True,
    )
