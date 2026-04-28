from fastapi import APIRouter, Depends, status
from sqlmodel import Session

from services.backend.routes.deps import get_current_user, get_session
from services.backend.schemas.feedback import FeedbackCreate, FeedbackCreateResponse
from services.backend.services import feedback_service
from services.backend.sqlDB.users import User

router = APIRouter(prefix="/feedback", tags=["feedback"])


@router.post("", response_model=FeedbackCreateResponse, status_code=status.HTTP_201_CREATED)
def create_feedback(
    payload: FeedbackCreate,
    session: Session = Depends(get_session),
    _: User = Depends(get_current_user),
):
    feedback = feedback_service.create_feedback(session, payload)
    return FeedbackCreateResponse(id=feedback.id)
