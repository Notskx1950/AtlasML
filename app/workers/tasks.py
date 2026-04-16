"""RQ task definitions — synchronous functions executed by workers."""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from typing import Any

import structlog

logger = structlog.get_logger()


def run_predict(
    job_id: str, model_name: str, version: str, inputs: list[dict]
) -> dict[str, Any]:
    """Run a prediction job synchronously in an RQ worker."""
    import asyncio

    from app.db.session import get_sync_session_factory
    from app.models.registry_store import RegistryStore, _build_adapter
    from app.db.models import JobRecord, ModelVersion

    db = get_sync_session_factory()()
    try:
        # Mark running
        record = db.get(JobRecord, uuid.UUID(job_id))
        if record:
            record.status = "running"
            db.commit()

        # Load model
        from sqlalchemy import select

        stmt = select(ModelVersion).where(
            ModelVersion.name == model_name, ModelVersion.version == version
        )
        mv = db.execute(stmt).scalar_one()
        adapter = _build_adapter(mv.artifact_uri, mv.runtime_config)

        # Run predict
        loop = asyncio.new_event_loop()
        predictions = loop.run_until_complete(adapter.predict(inputs))
        loop.close()

        result = {"predictions": predictions, "model_version": version}

        # Mark succeeded
        if record:
            record.status = "succeeded"
            record.result = result
            record.finished_at = datetime.now(timezone.utc)
            db.commit()

        logger.info("job_predict_succeeded", job_id=job_id, model=model_name)
        return result

    except Exception as exc:
        if record:
            record.status = "failed"
            record.error_msg = str(exc)
            record.finished_at = datetime.now(timezone.utc)
            db.commit()
        logger.error("job_predict_failed", job_id=job_id, error=str(exc))
        raise
    finally:
        db.close()


def run_eval(
    run_id: str, model_name: str, version: str, dataset_path: str
) -> dict[str, float]:
    """Run an evaluation job synchronously in an RQ worker."""
    from sqlalchemy import select

    from app.db.session import get_sync_session_factory
    from app.db.models import EvalRun, ModelVersion
    from app.eval.runner import EvalRunner
    from app.models.registry_store import _build_adapter

    db = get_sync_session_factory()()
    try:
        # Load model
        stmt = select(ModelVersion).where(
            ModelVersion.name == model_name, ModelVersion.version == version
        )
        mv = db.execute(stmt).scalar_one()
        adapter = _build_adapter(mv.artifact_uri, mv.runtime_config)

        metrics = EvalRunner.run_sync(run_id, adapter, dataset_path, db)
        logger.info("job_eval_succeeded", run_id=run_id, metrics=metrics)
        return metrics

    except Exception as exc:
        import uuid as uuid_mod

        run = db.get(EvalRun, uuid_mod.UUID(run_id))
        if run:
            run.status = "failed"
            run.finished_at = datetime.now(timezone.utc)
            db.commit()
        logger.error("job_eval_failed", run_id=run_id, error=str(exc))
        raise
    finally:
        db.close()
