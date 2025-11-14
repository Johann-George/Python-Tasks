from fastapi import FastAPI, File, UploadFile, Request, HTTPException, Response, status
from datetime import datetime
import traceback
import sys
import os
import json
import logging
from typing import Union
from model.models import Ocr_Status

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from Day_7_OCR.main import OCR
from crud.ocr import OCRService

app = FastAPI()
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(message)s')

@app.post("/api/v1/ocr/store")
async def extract_df(file: UploadFile = File(...)):
    try:
        logging.info("Post API invoked")
        upload_dir = "uploads"
        os.makedirs(upload_dir, exist_ok=True)
        file_path = os.path.join(upload_dir, file.filename)
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
            table_start_row=17,
            refine_data=True  
        )
        
        df, extractor = pipeline.run()
        OCRService.create_ocr(file_name=file.filename, ocr_data=df.to_json(orient='records'), ocr_status=Ocr_Status("Success"), created_at=datetime.now())
        logging.info("Successfully stored the value in the db")
        return {"message":"Successfully stored the data in the db!"}

    except Exception as e:
        OCRService.create_ocr(file_name=file.filename, ocr_data=df.to_json(orient='records'), ocr_status=Ocr_Status("Failed"), created_at=datetime.now())
        print("An unexpected error occured main():",e)
        traceback.print_exc()

    finally:
        os.remove(file_path)

@app.get("/api/v1/ocr/")
async def get_details(response: Response, id: int | None = None):
    logging.info("Get API invoked!")
    query = OCRService.get_ocr(id)
    if not query:
        raise HTTPException(status_code=404, detail="Invalid ID entered")
    return query

@app.put("/api/v1/update_ocr/")
async def update_details(request: Request, id: int | None = None ):
    try:
        logging.info("Put API invoked")
        raw_body = await request.body()
        if not raw_body: 
            raise HTTPException(status_code=500, detail="Please provide a valid Dataframe")
        json_string = raw_body.decode("utf-8")
        data = json.loads(json_string)
        logging.info("Retrieved the data from the request body")
        return OCRService.update_ocr(id, data)
    except json.JSONDecodeError as e:
        raise HTTPException(status_code=500, detail="Please provide a valid Dataframe")

@app.delete("/api/v1/delete_ocr/", response_model=dict)
async def delete_details(id: int | None = None):
    logging.info("Delete API invoked")
    OCRService.delete_ocr(id)
    logging.info("Data deleted successfully")
    return {"detail": "Record successfully deleted"}