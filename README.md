# Liver Tumor Segmentation with LLM Response

## Overview
This project implements a deep learning solution for liver tumor segmentation using an EfficientNet-UNet architecture, enhanced with explainability through SHAP (SHapley Additive exPlanations) values and LLM-generated medical explanations. The system includes a web interface for uploading medical images, performing segmentation, and viewing results with detailed medical insights.

## Project Structure
```
.
├── backend/               # Python backend server
│   ├── main.py            # API endpoints and server setup
│   └── ml_model.py        # Model loading and inference
├── frontend/              # Web interface
│   ├── css/               # Styling files
│   ├── js/                # JavaScript code
│   ├── static/            # Static assets
│   ├── index.html         # Landing page
│   ├── segmentation.html  # Segmentation interface
│   ├── history.html       # Past results
│   └── research.html      # Research information
├── models/                # Trained models
│   └── efficientnet_unet_model.h5  # Main segmentation model
├── Notebooks/             # Jupyter notebooks for development
├── test_images/           # Sample images for testing
├── db/                    # Database storage
├── docker-compose.yml     # Docker composition
├── Dockerfile             # Docker container configuration
└── requirements.txt       # Python dependencies
```

## Features
- Medical image segmentation of liver tumors
- Interactive web interface for uploading and analyzing images
- History tracking of previous segmentations
- Model explainability using SHAP values
- LLM-based medical interpretations of results


## Installation

### Local Setup
1. Clone the repository:
```bash
git clone https://github.com/yourusername/Liver-Tumor-Segmentation-with-LLM-Response.git
cd Liver-Tumor-Segmentation-with-LLM-Response
```

The code is optimized to run on docker and be hosted in Azure:


2. Build and run with Docker Compose:
```bash
docker-compose up --build
```

### Deployment
The application can be deployed to Azure using the provided scripts:
```bash
./deploy-to-azure.sh
```

To delete the deployment:
```bash
./delete-azure-deployment.sh
```


## Model Information
The segmentation model is based on an EfficientNet-UNet architecture trained on liver CT scans. The model processes medical images at a 128x128 resolution and produces pixel-wise segmentation masks highlighting tumor regions.

Key model specifications:

- Architecture: EfficientNet backbone with UNet-style decoder
- Input size: 128×128×3
- Target: Binary segmentation of tumor regions
- Multiple trained versions with different epochs (30, 55, etc.)

## Usage
1. Access the web interface
2. Upload a medical image (CT scan) through the segmentation page
3. View the segmentation results showing predicted tumor regions
4. Explore the model's explanation using SHAP values
5. Read the LLM-generated medical interpretation

## Dataset
The model was trained on a liver tumor segmentation dataset containing:
- CT scans in NII format
- Corresponding segmentation masks
- Data was preprocessed and split into training/validation/test sets


## Research and Development
The Jupyter notebooks in the Notebooks directory contain the research work, including:
- Data preprocessing and exploration
- Model architecture design
- Training processes
- Evaluation metrics
- SHAP value calculations for model explainability

# Liver Tumor Segmentation Application

This application provides liver tumor segmentation using deep learning models.

## Prerequisites

- Python 3.8+
- TensorFlow 2.x
- FastAPI
- Docker (for containerized deployment)

## Running the Application

The application can be run either directly on your host machine or using Docker.

### Running Normally (Without Docker)

1. Install the required dependencies:
   ```
   pip install fastapi uvicorn tensorflow pillow scipy
   ```

2. Modify the `main.py` file to use the local file system:
   - Uncomment the line: `sys.path.append(os.path.abspath("/media/epein5/Data/Liver-Tumor-Segmentation-with-LLM-Response/backend"))`
   - UnComment out the line: `model = tf.keras.models.load_model("models/efficientnet_unet_55ephocsss.h5")`
   - Ensure the model path is correctly set for your environment

3. Run the application:
   ```
   python backend/main.py
   ```

4. Access the application at http://localhost:8000

### Running with Docker

1. Modify the `main.py` file for Docker deployment:
   - Comment out the line: `sys.path.append(os.path.abspath("/media/epein5/Data/Liver-Tumor-Segmentation-with-LLM-Response/backend"))`
   - Uncomment the line: `model = tf.keras.models.load_model("models/efficientnet_unet_55ephocsss.h5")`

2. Build and run the  the Docker image:
   ```
   docker-compose up -d --build .
   ```


4. Access the application at http://localhost:8000

## Code Configuration for Deployment

The application requires specific configurations for different deployment methods. In `main.py`, you need to adjust these lines:

```python
# For normal local execution - uncomment this:
sys.path.append(os.path.abspath("/media/epein5/Data/Liver-Tumor-Segmentation-with-LLM-Response/backend"))

# For Docker execution - comment out the above and use this model loading approach:
# model = tf.keras.models.load_model("models/efficientnet_unet_55ephocsss.h5")
```

Make sure to adjust these settings based on your chosen deployment method.
