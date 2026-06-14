# Run (at root path): uvicorn web.app:app --reload

import os
import io
from PIL import Image
from fastapi import FastAPI, UploadFile, File, HTTPException, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from web.predictor import Predictor

app = FastAPI()

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))

# Configure Jinja2Templates for HTML responses
templates = Jinja2Templates(directory=CURRENT_DIR)

# Mount static files
app.mount("/static", StaticFiles(directory=CURRENT_DIR, html=True), name="static")

predictor = Predictor()

@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    return templates.TemplateResponse(request, "index.html", {"request": request})

@app.post("/predict", response_class=HTMLResponse)
async def predict(file: UploadFile = File(...)):
    if not file:
        raise HTTPException(status_code=400, detail="No file part in the request")
    if file.filename == '':
        raise HTTPException(status_code=400, detail="No selected file")

    try:
        image_bytes = await file.read()
        image = Image.open(io.BytesIO(image_bytes))
        
        predicted_text = predictor.predict(image)

        return JSONResponse(content={'prediction': predicted_text})
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

