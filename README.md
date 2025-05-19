#  Simple Object Detection Web Application

This project is a lightweight object detection web app built with **Streamlit**. It uses an ensemble of **YOLOv5** and **SSD (Single Shot Detector)** models to identify common objects (such as people, vehicles, animals, etc.) in uploaded images. Detected objects are shown with bounding boxes and labels directly on the image.



##  How the Application Works

1. The user uploads an image in JPG or PNG format via a Streamlit web interface.
2. The image is processed using two pre-trained models:
   - **YOLOv5** (loaded via `torch.hub`)
   - **SSD MobileNet V3** (from `torchvision`)
3. The predictions from both models are fused using ** Weighted Boxes Fusion (WBF)**.
4. A custom conflict resolution method suppresses overlapping boxes with different labels.
5. The final detected objects are drawn on the image using PIL and displayed in the interface, along with their confidence scores.
6. All the processed image will be store in static/uploads.

## Requirements & Dependencies

This application requires **Python 3.10 or later**. To install all required libraries, use the provided `requirements.txt`.

Main dependencies:
- `streamlit`
- `torch`
- `torchvision`
- `pillow`
- `numpy`
- `ensemble-boxes`
- `opencv-python` 

##  How to Set Up and Run the App
### Step 1: Clone 
Clone the GitHub repo to the local machine
### Step 2: Create and activate a virtual environment 
python -m venv detection
.\detection\Scripts\activate
### Step 3: Install requirements
pip install -r requirements.txt
### Step 4: Run the app
streamlit run app.py

### Some significant note
As I trying to deploy I faced the problem the streamlit not support YOLO5 or openCV so I decide to keep the ensemble model and run locally instead of 
removing the YOLO model -> in future using FastAPI and googlecould for improvement
