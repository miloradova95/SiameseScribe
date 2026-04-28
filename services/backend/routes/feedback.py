from fastapi import APIRouter, Depends
from sqlmodel import Session

from services.backend.routes.deps import get_current_user, get_session
from services.backend.schemas.feedback import FeedbackCreate, FeedbackCreateResponse, FeedbackListItem
from services.backend.services import feedback_service
from services.backend.sqlDB.users import User

router = APIRouter(prefix="/feedback", tags=["feedback"])


@router.get("/mine", response_model=list[FeedbackListItem])
def get_my_feedback(
    session: Session = Depends(get_session),
    current_user: User = Depends(get_current_user),
):
    return feedback_service.list_feedback_for_user(session, current_user)


@router.get("/mine/by-pair", response_model=FeedbackListItem | None)
def get_my_feedback_for_pair(
    query_patch_id: int,
    result_patch_id: int,
    session: Session = Depends(get_session),
    current_user: User = Depends(get_current_user),
):
    return feedback_service.get_feedback_for_user_pair(
        session,
        current_user,
        query_patch_id,
        result_patch_id,
    )


@router.post("", response_model=FeedbackCreateResponse)
def create_feedback(
    payload: FeedbackCreate,
    session: Session = Depends(get_session),
    current_user: User = Depends(get_current_user),
):
    feedback = feedback_service.create_feedback(session, payload, current_user)
    return FeedbackCreateResponse(id=feedback.id)
