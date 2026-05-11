import json
import logging
import os
import time
from datetime import datetime, timezone

import httpx
from sqlmodel import Session, select

from services.backend.database import engine
from services.backend.services.image_service import _post_to_ml_api
from services.backend.sqlDB.feedback import Feedback
from services.backend.sqlDB.finetune_run import FinetuneRun
from services.backend.sqlDB.patches import Patch
from services.ML.app.services.Finetune import count_samples

ML_API_URL = os.getenv("ML_API_URL", "http://127.0.0.1:8001").rstrip("/")

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────

def _build_feedback_payload(session: Session, feedback_items: list[Feedback]) -> list[dict]:
    patch_ids = (
        {f.query_patch_id for f in feedback_items}
        | {f.result_patch_id for f in feedback_items}
    )
    patches = session.exec(select(Patch).where(Patch.id.in_(patch_ids))).all()
    patches_by_id = {p.id: p for p in patches}

    payload = []
    for f in feedback_items:
        q = patches_by_id.get(f.query_patch_id)
        r = patches_by_id.get(f.result_patch_id)
        if not q or not r:
            logger.warning("Skipping feedback %d: patch not found", f.id)
            continue
        payload.append({
            "query_patch_path": q.file_path,
            "result_patch_path": r.file_path,
            "is_similar": f.label == 1,
        })
    return payload


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _ensure_utc(dt: datetime | None) -> datetime | None:
    if dt is None:
        return None
    return dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)


# ─────────────────────────────────────────────
# Orchestration
# ─────────────────────────────────────────────

def evaluate_and_trigger(session: Session, trigger_source: str = "auto") -> int | None:
    """
    Evaluate whether a finetune run should start and, if so, create a FinetuneRun record.

    Returns the new FinetuneRun.id, or None if the run was skipped.
    The caller is responsible for dispatching run_automated_finetune_job(run_id)
    in a background thread.
    """
    # 1. Don't start if a run is already active in any in-progress state
    active = session.exec(
        select(FinetuneRun).where(
            FinetuneRun.status.in_(["pending", "running", "reembedding", "evaluating"])
        )
    ).first()
    if active:
        logger.debug("Finetune skipped: run %d is already %s", active.id, active.status)
        return None

    # 2. Cooldown check (auto only — manual trigger bypasses this)
    if trigger_source == "auto":
        last = session.exec(
            select(FinetuneRun)
            .where(FinetuneRun.status == "completed")
            .order_by(FinetuneRun.completed_at.desc())
        ).first()
        if last and last.completed_at:
            cooldown_secs = int(os.getenv("FINETUNE_COOLDOWN_MINUTES", "30")) * 60
            elapsed = (_now() - _ensure_utc(last.completed_at)).total_seconds()
            if elapsed < cooldown_secs:
                logger.debug(
                    "Finetune skipped: cooldown active (%.0fs remaining)",
                    cooldown_secs - elapsed,
                )
                return None

    # 3. Load all feedback not yet used for a run
    unused = session.exec(
        select(Feedback).where(Feedback.used_for_retrain == False)
    ).all()
    if not unused:
        logger.debug("Finetune skipped: no unused feedback")
        return None

    # 4. Check ML trigger condition
    payload = _build_feedback_payload(session, list(unused))
    if not payload:
        logger.debug("Finetune skipped: feedback produced no valid patch pairs")
        return None

    samples = count_samples(payload)
    if not samples["can_finetune"]:
        logger.info(
            "Finetune skipped: trigger not met (t_real=%d, t_aug=%d, p_pos=%d)",
            samples["t_real"], samples["t_aug"], samples["p_pos"],
        )
        return None

    # 5. Create run record and lock the feedback
    feedback_ids = [f.id for f in unused]
    run = FinetuneRun(
        trigger_source=trigger_source,
        t_real=samples["t_real"],
        t_aug=samples["t_aug"],
        p_pos=samples["p_pos"],
        feedback_ids=json.dumps(feedback_ids),
    )
    session.add(run)
    for f in unused:
        f.used_for_retrain = True
    session.add_all(unused)
    session.commit()
    session.refresh(run)

    logger.info(
        "Finetune run %d queued (source=%s, t_real=%d, t_aug=%d, p_pos=%d, feedback_items=%d)",
        run.id, trigger_source,
        samples["t_real"], samples["t_aug"], samples["p_pos"], len(feedback_ids),
    )
    return run.id


def run_automated_finetune_job(run_id: int) -> None:
    """
    Execute the ML /retrain call for the given FinetuneRun.
    Updates the run record on completion or failure.
    On failure the consumed feedback is reset to used_for_retrain=False so it
    can be picked up by the next scheduled check.
    """
    logger.info("Finetune run %d: starting", run_id)
    feedback_ids: list[int] = []

    # Mark as running and build the payload
    with Session(engine) as session:
        run = session.get(FinetuneRun, run_id)
        if run is None:
            logger.error("Finetune run %d not found in DB", run_id)
            return

        run.status = "running"
        run.started_at = _now()
        session.add(run)
        session.commit()

        feedback_ids = json.loads(run.feedback_ids)
        feedback_items = session.exec(
            select(Feedback).where(Feedback.id.in_(feedback_ids))
        ).all()
        payload = _build_feedback_payload(session, list(feedback_items))

    if not payload:
        _mark_failed(run_id, feedback_ids, "No valid patch pairs in feedback")
        return

    try:
        response = _post_to_ml_api(
            "/retrain",
            json={"feedback": payload, "k_triplets": 1},
            timeout=60.0 * 60.0,
        )
        result = response.json()

        mlflow_run_id = result.get("run_id")

        with Session(engine) as session:
            run = session.get(FinetuneRun, run_id)
            run.status = "reembedding"
            run.mlflow_run_id = mlflow_run_id
            run.triplets_used = result.get("triplets_used", 0)
            run.t_real = result.get("t_real", run.t_real)
            run.t_aug  = result.get("t_aug",  run.t_aug)
            run.p_pos  = result.get("p_pos",  run.p_pos)
            run.reembedding_started_at = _now()
            session.add(run)
            session.commit()

        logger.info("Finetune run %d: ML training done, reloading model", run_id)
        _reload_model()

        logger.info("Finetune run %d: starting re-embedding", run_id)
        _start_reembed(mlflow_run_id)

        # Poll until re-embedding + eval completes (max 4 hours)
        eval_precision, eval_map = _poll_reembed_status(run_id, timeout_secs=4 * 3600)

        with Session(engine) as session:
            run = session.get(FinetuneRun, run_id)
            run.status = "completed"
            run.completed_at = _now()
            run.reembedding_completed_at = _now()
            run.eval_precision_at_k = eval_precision
            run.eval_mAP = eval_map
            session.add(run)
            session.commit()

        logger.info(
            "Finetune run %d: completed (mlflow=%s, triplets=%d, P@5=%.4f, mAP=%.4f)",
            run_id, mlflow_run_id, result.get("triplets_used", 0),
            eval_precision or 0, eval_map or 0,
        )

    except (httpx.HTTPError, Exception) as exc:
        logger.exception("Finetune run %d: failed — %s", run_id, exc)
        _mark_failed(run_id, feedback_ids, str(exc)[:500])


def _reload_model() -> None:
    try:
        with httpx.Client(timeout=30.0) as client:
            r = client.post(f"{ML_API_URL}/reload_model")
            r.raise_for_status()
        logger.info("Model weights reloaded in ML service")
    except Exception:
        logger.exception("Failed to reload model weights — ML service may be using stale weights")


def _start_reembed(mlflow_run_id: str | None) -> None:
    try:
        with httpx.Client(timeout=30.0) as client:
            r = client.post(
                f"{ML_API_URL}/reembed_all",
                json={"mlflow_run_id": mlflow_run_id},
            )
            if r.status_code == 409:
                # Another re-embedding is already running — piggyback on it by polling
                logger.warning("Re-embedding already in progress in ML service; will poll existing run")
                return
            r.raise_for_status()
        logger.info("Re-embedding started in ML service")
    except httpx.HTTPStatusError:
        logger.exception("Failed to start re-embedding")
        raise


def _poll_reembed_status(run_id: int, timeout_secs: int = 14400) -> tuple[float | None, float | None]:
    """Poll GET /reembed_status until in_progress=False. Returns (precision_at_k, mAP)."""
    poll_interval = 30
    elapsed = 0
    while elapsed < timeout_secs:
        time.sleep(poll_interval)
        elapsed += poll_interval
        try:
            with httpx.Client(timeout=15.0) as client:
                r = client.get(f"{ML_API_URL}/reembed_status")
                r.raise_for_status()
                status = r.json()
            if not status.get("in_progress", True):
                return status.get("eval_precision_at_k"), status.get("eval_mAP")
            logger.debug("Finetune run %d: re-embedding still in progress (%ds elapsed)", run_id, elapsed)
        except Exception:
            logger.exception("Error polling reembed_status for run %d", run_id)
    logger.warning("Finetune run %d: re-embedding polling timed out after %ds", run_id, timeout_secs)
    return None, None


def _mark_failed(run_id: int, feedback_ids: list[int], error_msg: str) -> None:
    with Session(engine) as session:
        # Release the feedback so it's available for the next run
        if feedback_ids:
            items = session.exec(select(Feedback).where(Feedback.id.in_(feedback_ids))).all()
            for f in items:
                f.used_for_retrain = False
            session.add_all(items)

        run = session.get(FinetuneRun, run_id)
        if run:
            run.status = "failed"
            run.completed_at = _now()
            run.error_msg = error_msg
            session.add(run)

        session.commit()
    logger.warning("Finetune run %d marked as failed: %s", run_id, error_msg)
