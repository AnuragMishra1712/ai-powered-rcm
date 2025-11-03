from sqlmodel import SQLModel, Field
from sqlalchemy import Column, ForeignKey
from sqlalchemy.dialects.postgresql import JSONB
from typing import Optional
import uuid
from datetime import datetime

class Prediction(SQLModel, table=True):
    __tablename__ = "predictions"

    id: uuid.UUID = Field(default_factory=uuid.uuid4, primary_key=True)
    user_id: Optional[uuid.UUID] = Field(default=None, foreign_key="users.id")

    feature_name: str
    input_json: dict = Field(default={}, sa_column=Column(JSONB))
    output_json: dict = Field(default={}, sa_column=Column(JSONB))
    created_at: datetime = Field(default_factory=datetime.utcnow)
