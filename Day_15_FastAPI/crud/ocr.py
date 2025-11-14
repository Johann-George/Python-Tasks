import json
import traceback
from enum import Enum
from fastapi import HTTPException
from datetime import datetime
from typing import List
from model.models import OCR
from config.db import db
from model.models import Ocr_Status, ocr_data


class OCRService:

    @staticmethod
    def create_ocr(file_name: str, ocr_data: json, ocr_status: Ocr_Status = None, created_at: datetime = None, updated_at: datetime = None):
        try:
            with db.get_session() as session:
                print("="*80)
                print("Status:", ocr_status.value)
                new_ocr = OCR(file_name=file_name, ocr_data=ocr_data, ocr_status=ocr_status.value, created_at=created_at, updated_at=updated_at)
                session.add(new_ocr)
                session.commit()
                session.refresh(new_ocr)
                return new_ocr
        except Exception as e:
            print("An unexpected error occured create_ocr():",e)
            traceback.print_exc()

    @staticmethod
    def get_ocr(ocr_id: int = None):
        try:
            with db.get_session() as session:
                if ocr_id:
                    return session.query(OCR).filter(OCR.id == ocr_id).first()
                return session.query(OCR).all()
        except Exception as e:
            print("An unexpected error occured get_ocr():", e)

    @staticmethod
    def delete_ocr(ocr_id: int):
        with db.get_session() as session:
            if not ocr_id:
                raise HTTPException(status_code=422, detail="Please provide an ID")
            ocr_obj = session.query(OCR).filter(OCR.id == ocr_id).first()
            if not ocr_obj:
                raise HTTPException(status_code=404, detail="OCR not found")
            session.delete(ocr_obj)
            session.commit()
            return False

    @staticmethod
    def update_ocr(ocr_id: int, ocr_data: List[ocr_data]):
        with db.get_session() as session:
            if not ocr_id:
                raise HTTPException(status_code=404, detail="Please provide an ID")
            ocr = session.query(OCR).filter(OCR.id == ocr_id).first()
            if not ocr:
                raise HTTPException(status_code=404, detail="ocr entry not found")
            ocr.ocr_data = str(ocr_data)
            ocr.updated_at = datetime.now()
            session.commit()
            session.refresh(ocr)
            return {"detail": "Row updated successfully"}