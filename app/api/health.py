from fastapi import APIRouter, Depends, status
from fastapi.responses import JSONResponse
from redis import Redis
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.deps import get_db
from app.config import settings

router = APIRouter()


@router.get("/health")
def health_check() -> dict[str, str]:
    return {"status": "ok"}


@router.get("/ready")
async def readiness_check(
    db: AsyncSession = Depends(get_db),
) -> JSONResponse:
    result = {
        "api": "ok",
        "postgres": "ok",
        "redis": "ok",
    }

    status_code = status.HTTP_200_OK

    try:
        await db.execute(text("SELECT 1"))
    except Exception:
        result["postgres"] = "error"
        status_code = status.HTTP_503_SERVICE_UNAVAILABLE

    try:
        redis_conn = Redis.from_url(settings.REDIS_URL)
        redis_conn.ping()
    except Exception:
        result["redis"] = "error"
        status_code = status.HTTP_503_SERVICE_UNAVAILABLE

    return JSONResponse(status_code=status_code, content=result)