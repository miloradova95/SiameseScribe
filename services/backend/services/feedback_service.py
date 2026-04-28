from fastapi import HTTPException
from sqlmodel import Session

from services.backend.schemas.feedback import FeedbackCreate, FeedbackLabel
from services.backend.sqlDB.feedback import Feedback
from services.backend.sqlDB.patches import Patch


def create_feedback(session: Session, payload: FeedbackCreate) -> Feedback:
    query_patch = session.get(Patch, payload.query_patch_id)
    if not query_patch:
        raise HTTPException(status_code=404, detail="Query patch not found")

    result_patch = session.get(Patch, payload.result_patch_id)
    if not result_patch:
        raise HTTPException(status_code=404, detail="Result patch not found")

    feedback = Feedback(
        query_patch_id=payload.query_patch_id,
        result_patch_id=payload.result_patch_id,
        label=_encode_label(payload.label),
    )
    session.add(feedback)
    session.commit()
    session.refresh(feedback)
    return feedback


def _encode_label(label: FeedbackLabel) -> int:
    return 1 if label == FeedbackLabel.similar else 0
