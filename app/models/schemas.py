"""
Pydantic request / response schemas for all API endpoints.
"""

from datetime import datetime

from pydantic import BaseModel, Field


# ── Upload ──────────────────────────────────────────────────────────────────

class UploadResponse(BaseModel):
    document_id: str
    filename: str
    total_chunks: int
    detected_language: str
    message: str = "Document uploaded and indexed successfully."


# ── Ask / Query ─────────────────────────────────────────────────────────────

class AskRequest(BaseModel):
    question: str = Field(..., min_length=1, max_length=2000)


class AskResponse(BaseModel):
    question: str
    answer: str
    detected_language: str
    source_documents: list[dict] = []
    routed_to: str = "rag"  # "rag" | "servicenow"


# ── Documents ───────────────────────────────────────────────────────────────

class DocumentInfo(BaseModel):
    document_id: str
    filename: str
    detected_language: str
    upload_timestamp: str
    chunk_count: int


class DocumentListResponse(BaseModel):
    documents: list[DocumentInfo]
    total: int


class DeleteResponse(BaseModel):
    document_id: str
    message: str = "Document deleted successfully."


# ── ServiceNow ──────────────────────────────────────────────────────────────

class HostRequest(BaseModel):
    host: str = Field(..., min_length=1, max_length=255)


class HostResponse(BaseModel):
    name: str = ""
    ip_address: str = ""
    os: str = ""
    location: str = ""
    install_status: str = ""


class ServiceNowErrorResponse(BaseModel):
    message: str = "No host found in ServiceNow CMDB."


# ── Chunk metadata (internal) ──────────────────────────────────────────────

class ChunkMetadata(BaseModel):
    document_id: str
    filename: str
    page_number: int
    detected_language: str
    upload_timestamp: str
    chunk_index: int
    text: str
