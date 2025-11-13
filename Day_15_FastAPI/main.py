from fastapi import FastAPI, File, UploadFile, Request
from datetime import datetime
import traceback
import sys
import os
import json
from typing import List
from model.models import Ocr_Status

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from Day_7_OCR.main import OCR
from crud.ocr import OCRService
from model.models import ocr_data

app = FastAPI()

@app.get("/")
async def greet():
    return {"message": "Hello!"}

@app.post("/api/v1/ocr/store")
async def extract_df(file: UploadFile = File(...)):
    try:
        upload_dir = "uploads"
        os.makedirs(upload_dir, exist_ok=True)
        file_path = os.path.join(upload_dir, file.filename)
        print(file_path)
        with open(file_path, "wb") as buffer:
            while content := await file.read(1024 * 1024):
                buffer.write(content)

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
        print("?"*80)
        print(df.to_json(orient='records'))
        OCRService.create_ocr(file_name=file.filename, ocr_data=df.to_json(orient='records'), ocr_status=Ocr_Status("Success"), created_at=datetime.now())
        return {"message":"Successfully updated the data to the db!"}

    except Exception as e:
        print("An unexpected error occured main():",e)
        traceback.print_exc()

    finally:
        os.remove(file_path)

@app.get("/api/v1/ocr/{id}")
async def get_details(id: int):
    try:
        return  OCRService.get_ocr(id)
    except Exception as e:
        print("An error occured get_details():", e)

@app.put("/api/v1/ocr/{id}")
async def update_details(id: int, request: Request):
    try:
        raw_body = await request.body()
        json_string = raw_body.decode("utf-8")
        data = json.loads(json_string)
        print("?"*80)
        print(data)
        OCRService.update_ocr(id, data)
    except Exception as e:
        print("An error occured update_details():",e)

@app.delete("/api/v1/ocr/{id}")
async def delete_details(id: int):
    try:
        OCRService.delete_ocr(id)
        return {"message": "Row successfully deleted"}
    except Exception as e:
        print("An error occured delete_details():", e)