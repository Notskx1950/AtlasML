from fastapi import APIRouter

router = APIRouter(prefix="/health")


@router.get("")
def health_check() -> dict[str, str]:
    """Health check endpoint."""
    return {"status": "ok"}
