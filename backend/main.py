import sys
import os
import shutil

# Add the backend directory to sys.path
# Comment if you want to build a docker image
sys.path.append(os.path.abspath("/media/epein5/Data/Liver-Tumor-Segmentation-with-LLM-Response/backend"))

from fastapi import FastAPI, File, UploadFile, HTTPException, Request, Form
from fastapi.responses import HTMLResponse, JSONResponse, RedirectResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
import base64
import uuid
from datetime import datetime
import uvicorn
from io import BytesIO
from PIL import Image
import tensorflow as tf
import scipy.ndimage
import json
from ml_model import analyze_and_save_medical_image
from fastapi.middleware.cors import CORSMiddleware
import numpy as np
import pandas as pd

# Import classification components
from classification import ClassificationRequest, process_classification

# Load the TensorFlow model

app = FastAPI(title="Liver Cancer Segmentation API")

model = tf.keras.models.load_model("models/efficientnet_unet_55ephocsss.h5", compile=False)


# Serve static files (CSS, JS)
app.mount("/static", StaticFiles(directory="./frontend"), name="static")

# Templates for HTML rendering
templates = Jinja2Templates(directory="./frontend")

os.makedirs("db", exist_ok=True)
os.makedirs("classificaltion", exist_ok=True)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins (you can restrict this to specific domains)
    allow_credentials=True,
    allow_methods=["*"],  # Allow all HTTP methods
    allow_headers=["*"],  # Allow all headers
)

@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.get("/segmentation", response_class=HTMLResponse)
async def segmentation_page(request: Request):
    return templates.TemplateResponse("segmentation.html", {"request": request})

@app.get("/classification", response_class=HTMLResponse)
async def classification_page(request: Request):
    """Serve the classification page"""
    return templates.TemplateResponse("classification.html", {"request": request})

@app.post("/classify")
async def classify_data(data: ClassificationRequest):
    """Process classification request and return results"""
    try:
        return process_classification(data)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Classification error: {str(e)}")

@app.get("/history", response_class=HTMLResponse)
async def history_page(request: Request):
    try:
        # Load all saved results from the database
        history = []
        for analysis_id in os.listdir("db"):
            analysis_dir = os.path.join("db", analysis_id)
            if os.path.isdir(analysis_dir):
                # Load metrics and explanation
                with open(os.path.join(analysis_dir, "tumor_metrics.json"), "r") as f:
                    tumor_metrics = json.load(f)
                with open(os.path.join(analysis_dir, "medical_report.txt"), "r") as f:
                    medical_explanation = f.read()

                # Add to history
                history.append({
                    "id": analysis_id,
                    "timestamp": datetime.fromtimestamp(os.path.getctime(analysis_dir)).strftime("%Y-%m-%d %H:%M:%S"),
                    "original_image": f"/results/{analysis_id}/original.png",
                    "segmented_image": f"/results/{analysis_id}/combined_segmentation.png",
                    "gradcam_image": f"/results/{analysis_id}/gradcam_tumor_focused.png",
                    "tumor_metrics": tumor_metrics,
                    "medical_explanation": medical_explanation,
                })

        # Sort history by timestamp (newest first)
        history.sort(key=lambda x: x["timestamp"], reverse=True)

        # Pass the history data to the template
        return templates.TemplateResponse("history.html", {"request": request, "history": history})
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/research", response_class=HTMLResponse)
async def research_page(request: Request):
    return templates.TemplateResponse("research.html", {"request": request})


def generate_explanation(tumor_size, confidence):
    """Generate a human-readable explanation based on tumor metrics."""
    if tumor_size < 5:
        severity = "minimal"
        recommendation = "routine follow-up may be recommended"
    elif tumor_size < 15:
        severity = "moderate"
        recommendation = "further diagnostic tests are recommended"
    else:
        severity = "significant"
        recommendation = "immediate consultation with a specialist is advised"
    
    confidence_level = "low" if confidence < 0.5 else "moderate" if confidence < 0.8 else "high"
    
    explanation = (
        f"The analysis indicates a {severity} tumor presence with {confidence_level} confidence. "
        f"The estimated tumor coverage is {tumor_size:.1f}mm in diameter. Based on these findings, {recommendation}. "
        f"Note that this is an AI-based assessment and should be confirmed by a medical professional."
    )
    
    return explanation


# API endpoint for image analysis
@app.post("/analyze")
async def analyze_image(file: UploadFile = File(...)):
    try:
        # Save the uploaded file temporarily
        file_path = f"temp/{uuid.uuid4()}.png"
        os.makedirs("temp", exist_ok=True)
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        # Analyze the image
        output_dir, tumor_metrics, medical_explanation = analyze_and_save_medical_image(
            model, file_path
        )

        # Prepare response
        response = {
            "output_dir": output_dir,
            "tumor_metrics": tumor_metrics,
            "medical_explanation": medical_explanation,
            "original_image_url": f"/results/{os.path.basename(output_dir)}/original.png",
            "segmented_image_url": f"/results/{os.path.basename(output_dir)}/combined_segmentation.png",
            "gradcam_image_url": f"/results/{os.path.basename(output_dir)}/gradcam_tumor_focused.png",
        }

        # Clean up temporary file
        os.remove(file_path)

        return response

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Serve result images
@app.get("/results/{analysis_id}/{filename}")
async def get_result_image(analysis_id: str, filename: str):
    file_path = f"db/{analysis_id}/{filename}"
    if os.path.exists(file_path):
        return FileResponse(file_path)
    else:
        raise HTTPException(status_code=404, detail="File not found")

@app.get("/history", response_class=HTMLResponse)
async def read_history():
    # Get all analysis results from the db directory
    history = []
    for analysis_id in os.listdir("db"):
        analysis_dir = os.path.join("db", analysis_id)
        if os.path.isdir(analysis_dir):
            # Load metrics and explanation
            with open(os.path.join(analysis_dir, "tumor_metrics.json"), "r") as f:
                tumor_metrics = json.load(f)
            with open(os.path.join(analysis_dir, "medical_report.txt"), "r") as f:
                medical_explanation = f.read()

            # Add to history
            history.append({
                "id": analysis_id,
                "timestamp": datetime.fromtimestamp(os.path.getctime(analysis_dir)).strftime("%Y-%m-%d %H:%M:%S"),
                "original_image": f"/results/{analysis_id}/original.png",
                "segmented_image": f"/results/{analysis_id}/combined_segmentation.png",
                "tumor_metrics": tumor_metrics,
                "medical_explanation": medical_explanation,
            })

    # Render the history template
    with open("history.html", "r") as file:
        content = file.read()
        content = content.replace("{% for item in history %}", "")
        content = content.replace("{% endfor %}", "")
        for item in history:
            content += f"""
            <div class="history-item">
                <img src="{item['original_image']}" alt="Original Image" class="history-item-thumbnail">
                <div class="history-item-info">
                    <h4>Analysis #{len(history) - history.index(item)}</h4>
                    <p><strong>Timestamp:</strong> {item['timestamp']}</p>
                    <p><strong>Tumor Metrics:</strong> {json.dumps(item['tumor_metrics'], indent=2)}</p>
                    <p><strong>Explanation:</strong> {item['medical_explanation']}</p>
                    <form action="/delete-history" method="post" style="display:inline;">
                        <input type="hidden" name="result_id" value="{item['id']}">
                        <button type="submit" class="delete-button">Delete</button>
                    </form>
                </div>
            </div>
            """
        return HTMLResponse(content=content)

# Delete a specific analysis
@app.post("/delete-history")
async def delete_history(result_id: str = Form(...)):
    try:
        analysis_dir = os.path.join("db", result_id)
        if os.path.exists(analysis_dir):
            shutil.rmtree(analysis_dir)
        return RedirectResponse(url="/history", status_code=303)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Clear all history
@app.post("/clear-history")
async def clear_history():
    try:
        for analysis_id in os.listdir("db"):
            analysis_dir = os.path.join("db", analysis_id)
            if os.path.isdir(analysis_dir):
                shutil.rmtree(analysis_dir)
        return RedirectResponse(url="/history", status_code=303)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    # import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)