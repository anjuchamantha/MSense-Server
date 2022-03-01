from typing import Optional
from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI()


class TestData(BaseModel):
    acc: str
    datetime: Optional[str] = None
    bat: str


@app.get("/")
async def root():
    return {"message": "Welcome to MSense-Server",
            "API Documentation": "/docs#/",
            "API Specification": "/redoc#/"}


@app.post("/test/{uid}")
async def test(uid: str, test_data: TestData):
    msg = "Test data received of the user: %s" % uid
    return {"message": msg,
            "test_data": test_data}

