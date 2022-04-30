from fastapi import FastAPI
from predict import predict_one

app = FastAPI()  # 创建API实例


@app.get("/")
async def detect_suicide():
    return {"tips": "轻生倾向检测:/SuicideDetector/xx"}


@app.get("/SuicideDetector/{sentence}")
async def detect_suicide(sentence: str):
    return {"label": predict_one(sentence)}
