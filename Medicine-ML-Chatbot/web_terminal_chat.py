import uvicorn
from fastapi import FastAPI, Request, Form
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
import os
import torch
import json
import numpy as np
from medquad_pytorch_intent_bot import predict_answer, model, accuracy

app = FastAPI()

# Templates & Static
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
templates = Jinja2Templates(directory=os.path.join(BASE_DIR, "templates"))

# Chatverlauf im Speicher (nur für Demo, nicht persistent)
chat_history = []
last_confidence = None

@app.get("/", response_class=HTMLResponse)
def index(request: Request):
    return templates.TemplateResponse("terminal_chat.html", {
        "request": request,
        "chat_history": chat_history,
        "accuracy": accuracy,
        "last_confidence": last_confidence,
    })

@app.post("/chat", response_class=HTMLResponse)
def chat(request: Request, user_input: str = Form(...)):
    global last_confidence
    answer, conf = predict_answer(user_input)
    chat_history.append((user_input, answer, conf))
    last_confidence = conf
    return RedirectResponse("/", status_code=303)

if __name__ == "__main__":
    uvicorn.run("web_terminal_chat:app", host="127.0.0.1", port=8000, reload=True) 