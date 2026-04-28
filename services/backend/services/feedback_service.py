from datetime import datetime, timezone
from fastapi import HTTPException
from pathlib import Path
from sqlmodel import Session, select

from services.backend.schemas.feedback import FeedbackCreate, FeedbackLabel, FeedbackListItem
from services.backend.sqlDB.feedback import Feedback
from services.backend.sqlDB.patches import Patch
from services.backend.sqlDB.users import User


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


def _encode_label(label: FeedbackLabel) -> int:
    return 1 if label == FeedbackLabel.similar else 0


def _decode_label(label: int) -> FeedbackLabel:
    return FeedbackLabel.similar if label == 1 else FeedbackLabel.not_similar


def _to_feedback_list_item(
    session: Session,
    feedback: Feedback,
    patches_by_id: dict[int, Patch] | None = None,
) -> FeedbackListItem:
    if patches_by_id is None:
        patches_by_id = _get_patches_by_id(session, [feedback])

    return FeedbackListItem(
        id=feedback.id,
        query_patch_id=feedback.query_patch_id,
        result_patch_id=feedback.result_patch_id,
        query_patch_file_name=_file_name_from_patch(patches_by_id.get(feedback.query_patch_id)),
        result_patch_file_name=_file_name_from_patch(patches_by_id.get(feedback.result_patch_id)),
        label=_decode_label(feedback.label),
        created_at=feedback.created_at,
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
