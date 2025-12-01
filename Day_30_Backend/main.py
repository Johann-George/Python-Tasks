import asyncio

from fastapi import FastAPI, Response, HTTPException

from Day_30_Backend.config.db import test_connection
from crud.flyjac_service import FlyjacService

app = FastAPI()

@app.get("/api/v1/get_data/")
async def get_data(response: Response, id: int | None = None):
    try:
        query = FlyjacService.get_info(id)
        if not query:
            raise HTTPException(status_code=404, detail="Invalid ID entered")
        return query
    except Exception as e:
        print("An unexpected error occurred get_data():",e)

if __name__ == "__main__":
    asyncio.run(test_connection())

