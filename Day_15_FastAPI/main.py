from fastapi import FastAPI, File, UploadFile
from datetime import datetime
import sys
import os
import httpx
from model.models import Status

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from Day_7_OCR.main import OCR
from crud.ocr import OCRService

app = FastAPI()

@app.get("/")
async def greet():
    return {"message": "Hello!"}

@app.post("/extract")
async def extract_df(file: UploadFile = File(...)):
    upload_dir = "uploads"
    os.makedirs(upload_dir, exist_ok=True)
    file_path = os.path.join(upload_dir, file.filename)
    pipeline = OCR(
        image_path=file_path,
        output_dir="outputs",
        preprocessing='advanced',
        tesseract_config='--psm 6 --oem 1',
        min_conf=0,
        show=False,
        # header_keywords=['NAME', 'CHILDREN', 'AGE', 'ADDRESS', 'GUARDIAN', 'PARENT'],
        table_start_row=17,
        refine_data=True  # Enable data refinement/cleaning
    )
    df, extractor = pipeline.run()
    OCRService.create_ocr(file_name=file.filename, ocr_data=df.to_json(orient='records'), status=Status.SUCCESS, created_at=datetime.now())
    return {"Successfully updated the data to the db!"}



     
