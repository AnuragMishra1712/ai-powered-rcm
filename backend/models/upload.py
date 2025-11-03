from sqlmodel import SQLModel, Field
from sqlalchemy import Column, ForeignKey
from sqlalchemy.dialects.postgresql import JSONB
from typing import Optional
import uuid
from datetime import datetime

class Upload(SQLModel, table=True):
    __tablename__ = "uploads"

    id: uuid.UUID = Field(default_factory=uuid.uuid4, primary_key=True)
    user_id: Optional[uuid.UUID] = Field(default=None, foreign_key="users.id")

    file_url: str
    extracted_text: Optional[str] = None
    prediction_result: Optional[dict] = Field(default=None, sa_column=Column(JSONB))
    created_at: datetime = Field(default_factory=datetime.utcnow)
