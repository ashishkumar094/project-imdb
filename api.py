from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
import joblib

app = FastAPI()

# Mount HTML folder
app.mount("/static", StaticFiles(directory="static"), name="static")

# Load your trained sentiment model pipeline
model = joblib.load("sentiment_model.pkl")

@app.get("/", response_class=HTMLResponse)
async def home():
    with open("static/index.html", "r", encoding="utf-8") as f:
        return f.read()

@app.post("/predict")
async def predict(request: Request):
    form = await request.form()
    review = form.get("review")

    prediction = model.predict([review])[0]

    result = "⭐ Positive Review" if prediction == 1 else "😞 Negative Review"

    return {"sentiment": result}

