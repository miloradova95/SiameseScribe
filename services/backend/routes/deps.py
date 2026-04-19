from sqlmodel import Session
from services.backend.database import engine

def get_session():
    with Session(engine) as session:
        yield session