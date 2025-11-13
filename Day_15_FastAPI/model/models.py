from enum import Enum as PyEnum
from sqlalchemy import Column, Integer, String, DateTime, Enum, Text
from sqlalchemy.sql import func
from config.db import db, Base
from typing import Optional
from pydantic import BaseModel, Field

class Ocr_Status(str, PyEnum):
    SUCCESS = 'Success'
    FAILED = 'Failed'

class ocr_data(BaseModel):
    No: str | None = None
    name_of_children: Optional[str] = Field(None, alias="Name of children")
    age: Optional[str] = Field(None, alias="Age") 
    address: Optional[str] = Field(None, alias="Address") 
    name_of_parent: Optional[str] = Field(None, alias="Name of guardian or parent")

class OCR(Base):
    __tablename__ = 'ocr'
    id = Column(Integer, primary_key=True, autoincrement=True)
    file_name = Column(String(400), nullable=False)
    ocr_data = Column(Text)
    ocr_status = Column(Enum(Ocr_Status), default=Ocr_Status("Success"))
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())

Base.metadata.create_all(db.get_engine())