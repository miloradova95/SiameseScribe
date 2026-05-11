import json

from fastapi import APIRouter, BackgroundTasks, Depends
from sqlmodel import Session, select

from services.backend.routes.deps import get_session, require_admin
from services.backend.schemas.finetune import FinetuneRunItem, FinetuneRunTriggerResponse
from services.backend.services import finetune_job
from services.backend.sqlDB.finetune_run import FinetuneRun
from services.backend.sqlDB.users import User

router = APIRouter(prefix="/admin/finetune-runs", tags=["finetune"])


@router.get("", response_model=list[FinetuneRunItem])
def list_finetune_runs(
    limit: int = 50,
    session: Session = Depends(get_session),
    _: User = Depends(require_admin),
):
    runs = session.exec(
        select(FinetuneRun).order_by(FinetuneRun.triggered_at.desc()).limit(limit)
    ).all()
    return [_to_item(r) for r in runs]


@router.post("/trigger", response_model=FinetuneRunTriggerResponse)
def trigger_finetune(
    background_tasks: BackgroundTasks,
    session: Session = Depends(get_session),
    _: User = Depends(require_admin),
):
    run_id = finetune_job.evaluate_and_trigger(session, trigger_source="manual")
    if run_id is None:
        return FinetuneRunTriggerResponse(
            status="skipped",
            reason="Conditions not met or a run is already active",
        )
    background_tasks.add_task(finetune_job.run_automated_finetune_job, run_id)
    return FinetuneRunTriggerResponse(status="triggered", run_id=run_id)


def _to_item(run: FinetuneRun) -> FinetuneRunItem:
    try:
        feedback_count = len(json.loads(run.feedback_ids))
    except Exception:
        feedback_count = 0
    return FinetuneRunItem(
        id=run.id,
        triggered_at=run.triggered_at,
        started_at=run.started_at,
        completed_at=run.completed_at,
        status=run.status,
        trigger_source=run.trigger_source,
        t_real=run.t_real,
        t_aug=run.t_aug,
        p_pos=run.p_pos,
        triplets_used=run.triplets_used,
        mlflow_run_id=run.mlflow_run_id,
        feedback_count=feedback_count,
        error_msg=run.error_msg,
        reembedding_started_at=run.reembedding_started_at,
        reembedding_completed_at=run.reembedding_completed_at,
        eval_precision_at_k=run.eval_precision_at_k,
        eval_mAP=run.eval_mAP,
    )
