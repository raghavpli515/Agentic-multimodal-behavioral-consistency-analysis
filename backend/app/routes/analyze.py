# This file defines the API route for analyzing uploaded videos. 
# It saves the uploaded video to a temporary location and then calls the processing pipeline to analyze it.
from fastapi import APIRouter, UploadFile, File
import shutil
import os

from app.services.pipeline_service import process_video

router = APIRouter() #Router is a component of FastAPI that allows you to organize your API endpoints. It helps in structuring the application by grouping related routes together. In this case, we are creating a router for handling video analysis requests.

UPLOAD_DIR = "backend/uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

@router.post("/analyze")  #This decorator defines a POST endpoint at the path "/analyze". When a client sends a POST request to this endpoint with a video file, the analyze_video function will be called to handle the request.
async def analyze_video(file: UploadFile = File(...)): #async fun is used to define an asynchronous function that can handle requests without blocking the server. UploadFile is a type provided by FastAPI that represents an uploaded file, and File(...) indicates that this parameter should be treated as a file upload.

    file_path = os.path.join(UPLOAD_DIR, file.filename)

    with open(file_path, "wb") as buffer:    #This line opens a new file in write-binary mode at the specified file path. The "with" statement ensures that the file is properly closed after the block of code is executed, even if an error occurs. The variable "buffer" is used to refer to this open file within the block.
        shutil.copyfileobj(file.file, buffer)

    result = process_video(file_path)

    return result