"""
ServiceNow CMDB integration service.

Queries the ``cmdb_ci_server`` table via the ServiceNow REST API using
async HTTP calls (httpx).
"""

from __future__ import annotations

import structlog
import httpx

from app.config import settings
from app.models.schemas import HostResponse

logger = structlog.get_logger(__name__)

CMDB_TABLE = "cmdb_ci_server"


class ServiceNowClient:
    """Async client for ServiceNow CMDB lookups."""

    def __init__(self) -> None:
        self._base_url: str = ""
        self._auth: tuple[str, str] = ("", "")

    def configure(self) -> None:
        self._base_url = settings.servicenow_base_url
        self._auth = (settings.servicenow_username, settings.servicenow_password)

    async def lookup_host(self, hostname: str) -> HostResponse | str:
        """
        Query CMDB for *hostname*.

        Returns a ``HostResponse`` on success or an error message string.
        """
        if not self._base_url or not self._auth[0]:
            return "ServiceNow is not configured. Set SERVICENOW_INSTANCE, SERVICENOW_USERNAME, and SERVICENOW_PASSWORD in .env."

        url = (
            f"{self._base_url}/api/now/table/{CMDB_TABLE}"
            f"?sysparm_query=name={hostname}"
            f"&sysparm_limit=1"
        )
        headers = {"Accept": "application/json"}

        try:
            async with httpx.AsyncClient(verify=False, timeout=30) as client:
                resp = await client.get(
                    url,
                    headers=headers,
                    auth=self._auth,
                )
                resp.raise_for_status()

            data = resp.json()
            results = data.get("result", [])

            if not results:
                return "No host found in ServiceNow CMDB."

            record = results[0]
            return HostResponse(
                name=record.get("name", ""),
                ip_address=record.get("ip_address", ""),
                os=record.get("os", ""),
                location=record.get("location", {}).get("display_value", "")
                if isinstance(record.get("location"), dict)
                else record.get("location", ""),
                install_status=record.get("install_status", ""),
            )

        except httpx.HTTPStatusError as exc:
            logger.error("servicenow_http_error", status=exc.response.status_code)
            return f"ServiceNow API error: {exc.response.status_code}"
        except Exception as exc:
            logger.error("servicenow_error", error=str(exc))
            return f"ServiceNow request failed: {exc}"


# Module-level singleton
servicenow_client = ServiceNowClient()
