from fastapi import FastAPI

app = FastAPI()


@app.get("/")
async def root():
    return {"message": "Hello World"}


@app.get("/test/{uid}")
async def test(uid: str):
    msg = "Test data received of the user: %s" % uid
    return {"message": msg}
