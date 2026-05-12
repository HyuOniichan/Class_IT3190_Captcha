import io
from PIL import Image
from fastapi import FastAPI, UploadFile, File, HTTPException, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from predictors.factory import build_captcha_predictor

app = FastAPI()

# Mount static files
app.mount("/static", StaticFiles(directory="static"), name="static")

# Configure Jinja2Templates for HTML responses
templates = Jinja2Templates(directory="templates")

predictor = build_captcha_predictor()

@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    return templates.TemplateResponse(request, "index.html", {"request": request})

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    if not file:
        raise HTTPException(status_code=400, detail="No file part in the request")
    if file.filename == '':
        raise HTTPException(status_code=400, detail="No selected file")

    try:
        image_bytes = await file.read()
        image = Image.open(io.BytesIO(image_bytes))
        
        predicted_text = predictor.predict_captcha(image)

        return JSONResponse(content={'prediction': predicted_text})
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
