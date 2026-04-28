from datetime import datetime, timezone
from pathlib import Path
from collections import defaultdict
import logging

import httpx
from fastapi import HTTPException
from sqlmodel import Session, select

from services.backend.database import engine
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


def start_retrain_job(
    session: Session,
    feedback_ids: list[int],
    _k_triplets: int,
) -> int:
    unique_feedback_ids = list(dict.fromkeys(feedback_ids))
    feedback_items = session.exec(
        select(Feedback).where(Feedback.id.in_(unique_feedback_ids))
    ).all()

    if len(feedback_items) != len(unique_feedback_ids):
        found_ids = {feedback.id for feedback in feedback_items}
        missing_ids = [feedback_id for feedback_id in unique_feedback_ids if feedback_id not in found_ids]
        raise HTTPException(status_code=404, detail=f"Feedback not found: {missing_ids}")

    busy_ids = [feedback.id for feedback in feedback_items if feedback.used_for_retrain]
    if busy_ids:
        raise HTTPException(
            status_code=409,
            detail=f"Feedback already in retrain job: {busy_ids}",
        )

    for feedback in feedback_items:
        feedback.used_for_retrain = True

    session.add_all(feedback_items)
    session.commit()

    return len(feedback_items)


def run_retrain_job(feedback_ids: list[int], k_triplets: int) -> None:
    unique_feedback_ids = list(dict.fromkeys(feedback_ids))
    try:
        with Session(engine) as session:
            feedback_items = session.exec(
                select(Feedback).where(Feedback.id.in_(unique_feedback_ids))
            ).all()
            if not feedback_items:
                return

            patches_by_id = _get_patches_by_id(session, feedback_items)
            payload = [
                {
                    "query_patch_path": _patch_path_or_404(patches_by_id, feedback.query_patch_id, "query"),
                    "result_patch_path": _patch_path_or_404(patches_by_id, feedback.result_patch_id, "result"),
                    "is_similar": _decode_label(feedback.label) == FeedbackLabel.similar,
                }
                for feedback in feedback_items
            ]

        triplets_used = _count_constructable_triplets(payload, k_triplets)
        if triplets_used == 0:
            logger.warning(
                "Skipping retrain because no constructable triplets were found",
                extra={"feedback_ids": unique_feedback_ids},
            )
            return

        response = _post_to_ml_api(
            "/retrain",
            json={"feedback": payload, "k_triplets": k_triplets},
            timeout=60.0 * 60.0,
        )
        result = response.json()
        logger.info(
            "Retrain job completed",
            extra={
                "feedback_ids": unique_feedback_ids,
                "triplets_used": result.get("triplets_used"),
                "run_id": result.get("run_id"),
            },
        )
    except httpx.HTTPError:
        logger.exception("Retrain job failed while calling the ML API", extra={"feedback_ids": unique_feedback_ids})
    except Exception:
        logger.exception("Retrain job failed", extra={"feedback_ids": unique_feedback_ids})
    finally:
        with Session(engine) as cleanup_session:
            feedback_items = cleanup_session.exec(
                select(Feedback).where(Feedback.id.in_(unique_feedback_ids))
            ).all()
            for feedback in feedback_items:
                feedback.used_for_retrain = False
            cleanup_session.add_all(feedback_items)
            cleanup_session.commit()


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


def _count_constructable_triplets(feedback_items: list[dict], k_triplets: int) -> int:
    buckets: dict[str, dict[str, list[str]]] = defaultdict(lambda: {"pos": [], "neg": []})
    for item in feedback_items:
        label_bucket = "pos" if item["is_similar"] else "neg"
        buckets[item["query_patch_path"]][label_bucket].append(item["result_patch_path"])

    total = 0
    for item in buckets.values():
        if item["pos"] and item["neg"]:
            total += k_triplets
    return total
