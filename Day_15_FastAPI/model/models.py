from enum import Enum as PyEnum
from sqlalchemy import Column, Integer, String, DateTime, Enum, Text
from sqlalchemy.dialects.mysql import JSON
from sqlalchemy.sql import func
from config.db import db, Base

class Status(PyEnum):
    SUCCESS = "Success"
    FAILED = "Failed"

class OCR(Base):
    __tablename__ = 'ocr'
    id = Column(Integer, primary_key=True, autoincrement=True)
    file_name = Column(String(400), nullable=False)
    ocr_data = Column(Text)
    status = Column(Enum(Status), default=Status.SUCCESS)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())


Base.metadata.create_all(db.get_engine())