import json
from datetime import datetime
from model.models import OCR
from config.db import db


class OCRService:

    @staticmethod
    def create_ocr(file_name: str, ocr_data: json, status: str = None, created_at: datetime = None, updated_at: datetime = None):
        try:
            with db.get_session() as session:
                new_ocr = OCR(file_name=file_name, ocr_data=ocr_data, status=status, created_at=created_at, updated_at=updated_at)
                session.add(new_ocr)
                session.commit()
                session.refresh(new_ocr)
                return new_ocr
        except Exception as e:
            print("An unexpected error occured:",e)
