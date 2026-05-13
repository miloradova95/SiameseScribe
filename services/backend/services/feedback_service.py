from datetime import datetime, timezone
from pathlib import Path
import logging

from fastapi import HTTPException
from sqlmodel import Session, select

from services.backend.schemas.feedback import (
    AdminFeedbackFilterParams,
    AdminFeedbackListItem,
    FeedbackCreate,
    FeedbackLabel,
    FeedbackListItem,
)
from services.backend.services.image_service import _post_to_ml_api
from services.backend.sqlDB.feedback import Feedback
from services.backend.sqlDB.patches import Patch
from services.backend.sqlDB.users import User

logger = logging.getLogger(__name__)


def create_feedback(session: Session, payload: FeedbackCreate, user: User) -> Feedback:
    query_patch = session.get(Patch, payload.query_patch_id)
    if not query_patch:
        raise HTTPException(status_code=404, detail="Query patch not found")

    result_patch = session.get(Patch, payload.result_patch_id)
    if not result_patch:
        raise HTTPException(status_code=404, detail="Result patch not found")

    feedback = session.exec(
        select(Feedback).where(
            Feedback.user_id == user.id,
            Feedback.query_patch_id == payload.query_patch_id,
            Feedback.result_patch_id == payload.result_patch_id,
        )
    ).first()

    if feedback:
        feedback.label = _encode_label(payload.label)
        feedback.used_for_retrain = False
        feedback.created_at = datetime.now(timezone.utc)
    else:
        feedback = Feedback(
            user_id=user.id,
            query_patch_id=payload.query_patch_id,
            result_patch_id=payload.result_patch_id,
            label=_encode_label(payload.label),
        )
        session.add(feedback)

    session.commit()
    session.refresh(feedback)
    return feedback


def get_feedback_for_user_pair(
    session: Session,
    user: User,
    query_patch_id: int,
    result_patch_id: int,
) -> FeedbackListItem | None:
    feedback = session.exec(
        select(Feedback).where(
            Feedback.user_id == user.id,
            Feedback.query_patch_id == query_patch_id,
            Feedback.result_patch_id == result_patch_id,
        )
    ).first()

    if not feedback:
        return None

    return _to_feedback_list_item(session, feedback)


def list_feedback_for_user(session: Session, user: User) -> list[FeedbackListItem]:
    feedback_items = session.exec(
        select(Feedback)
        .where(Feedback.user_id == user.id)
        .order_by(Feedback.created_at.desc())
    ).all()

    if not feedback_items:
        return []

    patches_by_id = _get_patches_by_id(session, feedback_items)

    return [_to_feedback_list_item(session, feedback, patches_by_id) for feedback in feedback_items]


def list_feedback_for_admin(
    session: Session,
    filters: AdminFeedbackFilterParams,
) -> list[AdminFeedbackListItem]:
    statement = select(Feedback, User).join(User, User.id == Feedback.user_id)

    if filters.user_id is not None:
        statement = statement.where(Feedback.user_id == filters.user_id)
    if filters.date_from is not None:
        statement = statement.where(Feedback.created_at >= filters.date_from)
    if filters.date_to is not None:
        statement = statement.where(Feedback.created_at <= filters.date_to)
    if filters.used_for_retrain is not None:
        statement = statement.where(Feedback.used_for_retrain == filters.used_for_retrain)

    rows = session.exec(statement.order_by(Feedback.created_at.desc())).all()
    if not rows:
        return []

    feedback_items = [feedback for feedback, _ in rows]
    patches_by_id = _get_patches_by_id(session, feedback_items)

    return [
        _to_admin_feedback_list_item(
            feedback,
            user,
            patches_by_id,
        )
        for feedback, user in rows
    ]


# start_retrain_job / run_retrain_job were removed here.
# Finetuning is now fully orchestrated by finetune_job.py:
#   - automatic: scheduler in main.py calls evaluate_and_trigger() every 15 min
#   - manual:    POST /admin/finetune-runs/trigger calls the same path
# Every run is logged in the FinetuneRun table. See services/backend/services/finetune_job.py.


def _encode_label(label: FeedbackLabel) -> int:
    return 1 if label == FeedbackLabel.similar else 0


def _decode_label(label: int) -> FeedbackLabel:
    return FeedbackLabel.similar if label == 1 else FeedbackLabel.not_similar


def _to_feedback_list_item(
    session: Session | None,
    feedback: Feedback,
    patches_by_id: dict[int, Patch] | None = None,
) -> FeedbackListItem:
    if patches_by_id is None:
        if session is None:
            raise ValueError("session is required when patches_by_id is not provided")
        patches_by_id = _get_patches_by_id(session, [feedback])

    return FeedbackListItem(
        id=feedback.id,
        query_patch_id=feedback.query_patch_id,
        result_patch_id=feedback.result_patch_id,
        query_patch_file_name=_file_name_from_patch(patches_by_id.get(feedback.query_patch_id)),
        result_patch_file_name=_file_name_from_patch(patches_by_id.get(feedback.result_patch_id)),
        label=_decode_label(feedback.label),
        created_at=feedback.created_at,
        used_for_retrain=feedback.used_for_retrain,
    )


def _to_admin_feedback_list_item(
    feedback: Feedback,
    user: User,
    patches_by_id: dict[int, Patch] | None = None,
) -> AdminFeedbackListItem:
    base_item = _to_feedback_list_item(None, feedback, patches_by_id)
    return AdminFeedbackListItem(
        **base_item.model_dump(),
        user_id=user.id,
        username=user.username,
        user_email=user.email,
    )


def _get_patches_by_id(session: Session, feedback_items: list[Feedback]) -> dict[int, Patch]:
    patch_ids = {
        feedback.query_patch_id
        for feedback in feedback_items
    } | {
        feedback.result_patch_id
        for feedback in feedback_items
    }
    patches = session.exec(select(Patch).where(Patch.id.in_(patch_ids))).all()
    return {patch.id: patch for patch in patches}


def _file_name_from_patch(patch: Patch | None) -> str:
    if not patch:
        return "Unknown patch"
    return Path(patch.file_path).name


def _patch_path_or_404(patches_by_id: dict[int, Patch], patch_id: int, patch_role: str) -> str:
    patch = patches_by_id.get(patch_id)
    if not patch:
        raise HTTPException(status_code=404, detail=f"{patch_role.title()} patch not found for feedback selection")
    return patch.file_path
