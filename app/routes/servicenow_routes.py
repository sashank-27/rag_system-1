"""
Direct ServiceNow CMDB lookup route.
"""

from __future__ import annotations

import structlog
from fastapi import APIRouter

from app.models.schemas import HostRequest, HostResponse, ServiceNowErrorResponse
from app.services.servicenow import servicenow_client

logger = structlog.get_logger(__name__)

router = APIRouter(prefix="/servicenow", tags=["ServiceNow"])


@router.post(
    "/host",
    response_model=HostResponse | ServiceNowErrorResponse,
)
async def lookup_host(body: HostRequest):
    """
    Query ServiceNow CMDB for a server by hostname.
    """
    logger.info("servicenow_lookup", host=body.host)
    result = await servicenow_client.lookup_host(body.host)

    if isinstance(result, str):
        return ServiceNowErrorResponse(message=result)

    return result
