from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse, FileResponse, StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from ultralytics import YOLO
from ultralytics.engine.results import Results, Boxes
from PIL import Image, ImageDraw
import io
import base64
import numpy as np
import logging
import time
import os
import json
from datetime import datetime
import shutil
import zipfile
from typing import List
import asyncio
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path


# Configure storage directories
STORAGE_BASE_DIR = "uploads"
LABELS_DIR = "labels"
MODELS_DIR = "models"
LOGS_DIR = "logs"
DETECTED_DIR = "detected"  # Directory for detected images with bboxes
CORRECTED_DIR = "corrected"  # Directory for corrected images
CORRECTED_LABELS_DIR = "corrected_labels"  # Directory for corrected labels


# Create directories if they don't exist
os.makedirs(STORAGE_BASE_DIR, exist_ok=True)
os.makedirs(LABELS_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(LOGS_DIR, exist_ok=True)
os.makedirs(DETECTED_DIR, exist_ok=True)
os.makedirs(CORRECTED_DIR, exist_ok=True)
os.makedirs(CORRECTED_LABELS_DIR, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/app.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


logger.info(f"Storage directories created")


def get_filename_mapping():
    """Get mapping between original filenames and timestamped filenames"""
    mapping = {}
    # Check uploads directory
    for filename in os.listdir(STORAGE_BASE_DIR):
        if '_' in filename:
            # Extract original filename from timestamped filename
            parts = filename.split('_', 2)
            if len(parts) >= 3:
                original_name = parts[2]
                mapping[original_name] = filename
    return mapping


def save_image_to_disk(image_data: bytes, filename: str) -> str:
    """Save image data to disk and return the saved file path"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    # Create unique filename to avoid conflicts
    name, ext = os.path.splitext(filename)
    unique_filename = f"{timestamp}_{name}{ext}"
    file_path = os.path.join(STORAGE_BASE_DIR, unique_filename)
   
    with open(file_path, "wb") as f:
        f.write(image_data)
   
    logger.info(f"Saved image to: {file_path}")
    return file_path


def save_detected_image(annotated_pil: Image.Image, filename: str) -> str:
    """Save detected image with bounding boxes to detected folder"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    name, ext = os.path.splitext(filename)
    detected_filename = f"{timestamp}_{name}_detected{ext}"
    detected_path = os.path.join(DETECTED_DIR, detected_filename)
   
    # Save the detected image
    annotated_pil.save(detected_path, format='JPEG', quality=95)
   
    logger.info(f"Saved detected image to: {detected_path}")
    return detected_path


def save_yolo_labels(detections: list, filename: str, img_width: int, img_height: int) -> str:
    """Save annotations in YOLO format for retraining"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    name, _ = os.path.splitext(filename)
    yolo_filename = f"{timestamp}_{name}.txt"
    yolo_path = os.path.join(LABELS_DIR, yolo_filename)
   
    with open(yolo_path, "w") as f:
        for detection in detections:
            # Convert to YOLO format (normalized coordinates)
            x_center = (detection["x"] + detection["width"] / 2) / img_width
            y_center = (detection["y"] + detection["height"] / 2) / img_height
            width = detection["width"] / img_width
            height = detection["height"] / img_height
           
            # Get class ID (assuming you have a class mapping)
            class_name = detection["className"]
            class_id = get_class_id(class_name)
           
            f.write(f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}\n")
   
    logger.info(f"Saved YOLO format labels to: {yolo_path}")
    return yolo_path


def get_class_id(class_name: str) -> int:
    """Map class name to class ID for YOLO format"""
    # You can customize this mapping based on your model's classes
    class_mapping = {name: idx for idx, name in model.names.items()}
    return class_mapping.get(class_name, 0)


def log_detections(detections: list, filename: str):
    """Log detections to labels.log file"""
    log_entry = {
        "timestamp": datetime.now().isoformat(),
        "filename": filename,
        "detections": detections,
        "total_detections": len(detections)
    }
   
    log_file_path = os.path.join(LOGS_DIR, "labels.log")
    with open(log_file_path, "a") as f:
        f.write(json.dumps(log_entry) + "\n")


app = FastAPI()


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Update this with your frontend URL in production
    allow_methods=["*"],
    allow_headers=["*"],
    allow_credentials=True,
)


# Load model with logging
try:
    logger.info("Loading YOLO model...")
    model = YOLO("models/yolov8.pt")
    logger.info("YOLO model loaded successfully")
except Exception as e:
    logger.error(f"Failed to load YOLO model: {e}")
    raise


@app.post("/api/label")
async def label_images(images: list[UploadFile] = File(...)):
    start_time = time.time()
    logger.info(f"Received request to label {len(images)} images")
   
    results = []
   
    for idx, image in enumerate(images):
        try:
            logger.info(f"Processing image {idx + 1}/{len(images)}: {image.filename}")
           
            # Read and validate image
            data = await image.read()
            logger.debug(f"Image size: {len(data)} bytes")
           
            if len(data) == 0:
                logger.warning(f"Empty image file: {image.filename}")
                continue
           
            # Save original image to disk
            original_image_path = save_image_to_disk(data, image.filename)
               
            img = Image.open(io.BytesIO(data)).convert("RGB")
            logger.debug(f"Image dimensions: {img.size}")
            img_width, img_height = img.size
           
            # Convert PIL to numpy for YOLO
            img_array = np.array(img)
           
            # Run YOLO prediction
            logger.debug(f"Running YOLO prediction on {image.filename}")
            pred_start = time.time()
            pred = model(img_array)[0]
            pred_time = time.time() - pred_start
            logger.debug(f"YOLO prediction completed in {pred_time:.2f} seconds")
           
            # Log detection count
            num_detections = len(pred.boxes) if pred.boxes is not None else 0
            logger.info(f"Found {num_detections} detections in {image.filename}")
           
            # Use YOLO's built-in visualization
            annotated_img = pred.plot()
           
            # Convert back to PIL Image
            annotated_pil = Image.fromarray(annotated_img)
           
            # Save detected image with timestamped filename
            detected_image_path = save_detected_image(annotated_pil, image.filename)
           
            # Convert to base64 for frontend
            img_buffer = io.BytesIO()
            annotated_pil.save(img_buffer, format='JPEG', quality=95)
            img_base64 = base64.b64encode(img_buffer.getvalue()).decode()
           
            # Extract detection data
            detections = []
            if pred.boxes is not None:
                for box, conf, cls in zip(pred.boxes.xyxy.cpu().numpy(),
                                        pred.boxes.conf.cpu().numpy(),
                                        pred.boxes.cls.cpu().numpy()):
                    x1, y1, x2, y2 = box
                    class_name = model.names[int(cls)]
                    confidence = round(float(conf), 2)
                   
                    detections.append({
                        "className": class_name,
                        "confidence": confidence,
                        "x": int(x1),
                        "y": int(y1),
                        "width": int(x2 - x1),
                        "height": int(y2 - y1),
                    })
                   
                    logger.debug(f"Detection: {class_name} (conf: {confidence}) at ({int(x1)}, {int(y1)})")
           
            # Save labels in YOLO format using timestamped filename
            yolo_label_path = save_yolo_labels(detections, image.filename, img_width, img_height)
           
            # Log detections to labels.log
            log_detections(detections, image.filename)
           
            results.append({
                "fileName": image.filename,
                "timestampedFileName": image.filename,
                "detections": detections,
                "imageWithBoxes": f"data:image/jpeg;base64,{img_base64}",
                "originalImagePath": original_image_path,
                "detectedImagePath": detected_image_path,
                "yoloLabelPath": yolo_label_path,
                "downloadUrl": f"/api/download/detected/{os.path.basename(detected_image_path)}",
                "imageWidth": img_width,
                "imageHeight": img_height
            })
           
            logger.info(f"Successfully processed and saved {image.filename}")
           
        except Exception as e:
            logger.error(f"Error processing image {image.filename}: {str(e)}")
            # Continue processing other images instead of failing completely
            results.append({
                "fileName": image.filename,
                "error": f"Failed to process: {str(e)}",
                "detections": [],
                "imageWithBoxes": None
            })
   
    total_time = time.time() - start_time
    total_detections = sum(len(result.get("detections", [])) for result in results)
   
    logger.info(f"Request completed in {total_time:.2f} seconds")
    logger.info(f"Total detections across all images: {total_detections}")
   
    return {"results": results}


@app.get("/api/download/detected/{filename}")
async def download_detected_image(filename: str):
    """Download a single detected image with bounding boxes"""
    file_path = os.path.join(DETECTED_DIR, filename)
   
    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="Detected image not found")
   
    logger.info(f"Downloading detected image: {filename}")
    return FileResponse(
        path=file_path,
        filename=filename,
        media_type='image/jpeg'
    )


@app.get("/api/download/original/{filename}")
async def download_original_image(filename: str):
    """Download a single original image"""
    file_path = os.path.join(STORAGE_BASE_DIR, filename)
   
    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="Original image not found")
   
    logger.info(f"Downloading original image: {filename}")
    return FileResponse(
        path=file_path,
        filename=filename,
        media_type='image/jpeg'
    )


@app.get("/api/download/batch/detected")
async def download_all_detected_images():
    """Download all detected images as a ZIP file"""
    try:
        # Create a ZIP file in memory
        zip_buffer = io.BytesIO()
       
        with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zip_file:
            # Add all detected images to the ZIP
            for filename in os.listdir(DETECTED_DIR):
                if filename.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.tiff')):
                    file_path = os.path.join(DETECTED_DIR, filename)
                    zip_file.write(file_path, filename)
       
        zip_buffer.seek(0)
       
        # Generate timestamp for unique filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        zip_filename = f"detected_images_{timestamp}.zip"
       
        logger.info(f"Created ZIP file with all detected images: {zip_filename}")
       
        return StreamingResponse(
            io.BytesIO(zip_buffer.read()),
            media_type="application/zip",
            headers={"Content-Disposition": f"attachment; filename={zip_filename}"}
        )
       
    except Exception as e:
        logger.error(f"Error creating ZIP file: {e}")
        raise HTTPException(status_code=500, detail=f"Error creating ZIP file: {str(e)}")


@app.post("/api/download/batch/selected")
async def download_selected_images(filenames: List[str]):
    """Download selected detected images as a ZIP file"""
    try:
        if not filenames:
            raise HTTPException(status_code=400, detail="No filenames provided")
       
        # Create a ZIP file in memory
        zip_buffer = io.BytesIO()
       
        with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zip_file:
            for filename in filenames:
                # Extract just the base filename from full path if needed
                base_filename = os.path.basename(filename)
                file_path = os.path.join(DETECTED_DIR, base_filename)
               
                if os.path.exists(file_path):
                    zip_file.write(file_path, base_filename)
                else:
                    logger.warning(f"File not found: {file_path}")
       
        zip_buffer.seek(0)
       
        # Generate timestamp for unique filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        zip_filename = f"selected_detected_images_{timestamp}.zip"
       
        logger.info(f"Created ZIP file with {len(filenames)} selected detected images: {zip_filename}")
       
        return StreamingResponse(
            io.BytesIO(zip_buffer.read()),
            media_type="application/zip",
            headers={"Content-Disposition": f"attachment; filename={zip_filename}"}
        )
       
    except Exception as e:
        logger.error(f"Error creating ZIP file for selected images: {e}")
        raise HTTPException(status_code=500, detail=f"Error creating ZIP file: {str(e)}")


@app.get("/api/list/detected")
async def list_detected_images():
    """Get list of all detected images available for download"""
    try:
        detected_files = []
        for filename in os.listdir(DETECTED_DIR):
            if filename.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.tiff')):
                file_path = os.path.join(DETECTED_DIR, filename)
                file_stats = os.stat(file_path)
               
                detected_files.append({
                    "filename": filename,
                    "size_bytes": file_stats.st_size,
                    "size_mb": round(file_stats.st_size / (1024 * 1024), 2),
                    "created_at": datetime.fromtimestamp(file_stats.st_ctime).isoformat(),
                    "download_url": f"/api/download/detected/{filename}"
                })
       
        # Sort by creation time (newest first)
        detected_files.sort(key=lambda x: x["created_at"], reverse=True)
       
        logger.info(f"Listed {len(detected_files)} detected images")
        return {"detected_images": detected_files}
       
    except Exception as e:
        logger.error(f"Error listing detected images: {e}")
        raise HTTPException(status_code=500, detail=f"Error listing detected images: {str(e)}")


@app.get("/api/storage-info")
async def get_storage_info():
    """Get information about stored files"""
    try:
        # Count files in each directory
        upload_count = len([f for f in os.listdir(STORAGE_BASE_DIR) if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.tiff'))])
        label_count = len([f for f in os.listdir(LABELS_DIR) if f.endswith('.txt')])
        detected_count = len([f for f in os.listdir(DETECTED_DIR) if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.tiff'))])
       
        # Get directory sizes
        def get_dir_size(directory):
            total_size = 0
            for dirpath, dirnames, filenames in os.walk(directory):
                for filename in filenames:
                    filepath = os.path.join(dirpath, filename)
                    total_size += os.path.getsize(filepath)
            return total_size
       
        storage_info = {
            "directories": {
                "uploads": {
                    "path": STORAGE_BASE_DIR,
                    "file_count": upload_count,
                    "size_mb": round(get_dir_size(STORAGE_BASE_DIR) / (1024 * 1024), 2)
                },
                "labels": {
                    "path": LABELS_DIR,
                    "file_count": label_count,
                    "size_mb": round(get_dir_size(LABELS_DIR) / (1024 * 1024), 2)
                },
                "detected": {
                    "path": DETECTED_DIR,
                    "file_count": detected_count,
                    "size_mb": round(get_dir_size(DETECTED_DIR) / (1024 * 1024), 2)
                }
            }
        }
       
        logger.info("Storage info requested")
        return storage_info
       
    except Exception as e:
        logger.error(f"Error getting storage info: {e}")
        raise HTTPException(status_code=500, detail=f"Error getting storage info: {str(e)}")


@app.delete("/api/clear-storage")
async def clear_storage():
    """Clear all stored files (use with caution!)"""
    try:
        import shutil
       
        # Clear uploads, labels, and detected directories
        shutil.rmtree(STORAGE_BASE_DIR)
        shutil.rmtree(LABELS_DIR)
        shutil.rmtree(DETECTED_DIR)
       
        # Recreate directories
        os.makedirs(STORAGE_BASE_DIR, exist_ok=True)
        os.makedirs(LABELS_DIR, exist_ok=True)
        os.makedirs(DETECTED_DIR, exist_ok=True)
       
        logger.warning("Storage directories cleared!")
        return {"message": "Storage cleared successfully"}
       
    except Exception as e:
        logger.error(f"Error clearing storage: {e}")
        raise HTTPException(status_code=500, detail=f"Error clearing storage: {str(e)}")


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    logger.info("Health check requested")
    return {"status": "healthy", "model_loaded": model is not None}


@app.on_event("startup")
async def startup_event():
    logger.info("FastAPI application starting up")


@app.on_event("shutdown")
async def shutdown_event():
    logger.info("FastAPI application shutting down")


def find_label_file(image_filename: str) -> str:
    """Helper function to find the corresponding label file"""
    logger.info(f"Searching for label file matching image: {image_filename}")
   
    def clean_filename(filename):
        """Clean filename for comparison"""
        # Remove any leading/trailing spaces
        filename = filename.strip()
        # Remove spaces around parentheses
        filename = filename.replace(' (', '(').replace(') ', ')')
        # Remove timestamp prefix if exists (format: YYYYMMDD_HHMMSS_)
        if '_' in filename:
            parts = filename.split('_', 2)
            if len(parts) >= 3:
                filename = parts[2]  # Get everything after the timestamp
        return filename
   
    def get_base_name(filename):
        """Get the base part of the filename without timestamp and extension"""
        # Remove extension
        filename = os.path.splitext(filename)[0]
        # Clean the filename
        return clean_filename(filename)
   
    # Get all label files
    label_files = os.listdir(LABELS_DIR)
    if not label_files:
        logger.error("No label files found in labels directory")
        return None
   
    # Get the base name of the image file (without timestamp and extension)
    image_base_name = get_base_name(image_filename)
    logger.info(f"Looking for matches for base name: {image_base_name}")
   
    # Find matching label files
    matching_files = []
    for label_file in label_files:
        label_base_name = get_base_name(label_file)
        if image_base_name == label_base_name:
            matching_files.append(label_file)
   
    if not matching_files:
        logger.error(f"No matching label file found for {image_filename}")
        logger.error(f"Image base name: {image_base_name}")
        logger.error(f"Available label files: {[get_base_name(f) for f in label_files]}")
        return None
   
    # If multiple matches found, get the most recent one
    if len(matching_files) > 1:
        logger.info(f"Multiple matches found: {matching_files}")
        matching_files.sort(key=lambda x: os.path.getctime(os.path.join(LABELS_DIR, x)), reverse=True)
        logger.info(f"Selected most recent: {matching_files[0]}")
   
    selected_file = os.path.join(LABELS_DIR, matching_files[0])
    logger.info(f"Selected label file: {selected_file}")
    return selected_file


@app.post("/api/update-label")
async def update_label(data: dict):
    """Update a label in the YOLO format labels file"""
    try:
        # Validate required fields
        required_fields = ['fileName', 'className', 'x', 'y', 'width', 'height']
        missing_fields = [field for field in required_fields if field not in data]
        if missing_fields:
            raise HTTPException(
                status_code=400,
                detail=f"Missing required fields: {', '.join(missing_fields)}"
            )
       
        filename = data['fileName']
        new_class = data['className']
       
        logger.info(f"Attempting to update label for {filename}")
        logger.info(f"Update data: {data}")
       
        # Find the corresponding label file
        label_file = find_label_file(filename)
        if not label_file:
            logger.error(f"No label file found for {filename}")
            raise HTTPException(status_code=404, detail=f"Label file not found for {filename}")
       
        logger.info(f"Found label file: {label_file}")
       
        # Read the current labels
        with open(label_file, 'r') as f:
            lines = f.readlines()
       
        # Get normalized coordinates from request if provided, otherwise calculate them
        if 'normalized_coords' in data:
            x_center = data['normalized_coords']['x_center']
            y_center = data['normalized_coords']['y_center']
            width = data['normalized_coords']['width']
            height = data['normalized_coords']['height']
        else:
            # Calculate normalized coordinates
            img_width = data.get('img_width', 1920)
            img_height = data.get('img_height', 1080)
            x_center = (data['x'] + data['width']/2) / img_width
            y_center = (data['y'] + data['height']/2) / img_height
            width = data['width'] / img_width
            height = data['height'] / img_height
       
        logger.info(f"Normalized coordinates: x={x_center}, y={y_center}, w={width}, h={height}")
       
        # Find and update the matching detection
        updated = False
        for i, line in enumerate(lines):
            parts = line.strip().split()
            if len(parts) == 5:
                # Compare the coordinates with some tolerance
                if (abs(float(parts[1]) - x_center) < 0.01 and
                    abs(float(parts[2]) - y_center) < 0.01 and
                    abs(float(parts[3]) - width) < 0.01 and
                    abs(float(parts[4]) - height) < 0.01):
                    # Update the class ID
                    class_id = get_class_id(new_class)
                    lines[i] = f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}\n"
                    updated = True
                    logger.info(f"Updated detection at line {i+1}")
                    logger.info(f"Old line: {line.strip()}")
                    logger.info(f"New line: {lines[i].strip()}")
                    break
       
        if not updated:
            logger.error(f"No matching detection found in {filename}")
            logger.error(f"Searched for coordinates: x={x_center}, y={y_center}, w={width}, h={height}")
            logger.error("Available detections:")
            for line in lines:
                logger.error(f"  {line.strip()}")
            raise HTTPException(
                status_code=404,
                detail="Detection not found in label file. Coordinates don't match any existing detection."
            )
       
        # Write back the updated labels
        with open(label_file, 'w') as f:
            f.writelines(lines)
       
        # Log the update
        log_entry = {
            "timestamp": datetime.now().isoformat(),
            "filename": filename,
            "action": "label_update",
            "old_class": data.get('oldClassName'),
            "new_class": new_class,
            "coordinates": {
                "x_center": x_center,
                "y_center": y_center,
                "width": width,
                "height": height
            }
        }
       
        with open(os.path.join(LOGS_DIR, "label_updates.log"), "a") as f:
            f.write(json.dumps(log_entry) + "\n")
       
        logger.info(f"Successfully updated label in {filename} from {data.get('oldClassName')} to {new_class}")
       
        return {
            "message": "Label updated successfully",
            "details": {
                "filename": filename,
                "old_class": data.get('oldClassName'),
                "new_class": new_class,
                "normalized_coords": {
                    "x_center": x_center,
                    "y_center": y_center,
                    "width": width,
                    "height": height
                }
            }
        }
       
    except HTTPException as he:
        logger.error(f"HTTP error in update_label: {he.detail}")
        raise he
    except Exception as e:
        logger.error(f"Unexpected error in update_label: {str(e)}")
        logger.exception(e)
        raise HTTPException(
            status_code=500,
            detail=f"Server error while updating label: {str(e)}"
        )


@app.post("/api/delete-detection")
async def delete_detection(data: dict):
    """Delete a detection from the YOLO format labels file and regenerate the image"""
    try:
        filename = data.get('fileName')
        original_confidence = data.get('confidence', 0.9)  # Get original confidence if available
       
        if not filename:
            raise HTTPException(status_code=400, detail="Missing filename")
       
        # Find the corresponding label file
        label_file = find_label_file(filename)
        if not label_file:
            raise HTTPException(status_code=404, detail=f"Label file not found for {filename}")
       
        logger.info(f"Found label file for deletion: {label_file}")
        logger.info(f"Received data: {json.dumps(data, indent=2)}")
       
        # Read the current labels
        with open(label_file, 'r') as f:
            lines = f.readlines()
            logger.info(f"Current labels in file: {lines}")
       
        # Get normalized coordinates from request if provided, otherwise calculate them
        if 'normalized_coords' in data:
            x_center = data['normalized_coords']['x_center']
            y_center = data['normalized_coords']['y_center']
            width = data['normalized_coords']['width']
            height = data['normalized_coords']['height']
            logger.info(f"Using provided normalized coordinates: x={x_center}, y={y_center}, w={width}, h={height}")
        else:
            # Calculate normalized coordinates
            img_width = data.get('img_width', 1920)
            img_height = data.get('img_height', 1080)
            x_center = (data['x'] + data['width']/2) / img_width
            y_center = (data['y'] + data['height']/2) / img_height
            width = data['width'] / img_width
            height = data['height'] / img_height
            logger.info(f"Calculated normalized coordinates: x={x_center}, y={y_center}, w={width}, h={height}")
            logger.info(f"From pixel coordinates: x={data['x']}, y={data['y']}, w={data['width']}, h={data['height']}")
            logger.info(f"Using image dimensions: width={img_width}, height={img_height}")
       
        # Find and remove the matching detection
        deleted = False
        new_lines = []
        remaining_detections = []
        for i, line in enumerate(lines):
            parts = line.strip().split()
            if len(parts) == 5:
                # Compare the coordinates with some tolerance
                current_x = float(parts[1])
                current_y = float(parts[2])
                current_w = float(parts[3])
                current_h = float(parts[4])
               
                # Log each comparison
                logger.info(f"Comparing line {i+1}:")
                logger.info(f"  Target:  x={x_center:.6f}, y={y_center:.6f}, w={width:.6f}, h={height:.6f}")
                logger.info(f"  Current: x={current_x:.6f}, y={current_y:.6f}, w={current_w:.6f}, h={current_h:.6f}")
                logger.info(f"  Differences: x={abs(current_x - x_center):.6f}, y={abs(current_y - y_center):.6f}, w={abs(current_w - width):.6f}, h={abs(current_h - height):.6f}")
               
                # Increase tolerance slightly
                tolerance = 0.02  # Increased from 0.01 to 0.02
                if (abs(current_x - x_center) < tolerance and
                    abs(current_y - y_center) < tolerance and
                    abs(current_w - width) < tolerance and
                    abs(current_h - height) < tolerance):
                    deleted = True
                    logger.info(f"Found matching detection to delete: {line.strip()}")
                    continue
                else:
                    # Keep track of remaining detections
                    remaining_detections.append({
                        "class_id": int(parts[0]),
                        "x_center": current_x,
                        "y_center": current_y,
                        "width": current_w,
                        "height": current_h,
                        "confidence": original_confidence  # Preserve original confidence
                    })
            new_lines.append(line)
       
        if not deleted:
            logger.error(f"No matching detection found to delete in {filename}")
            logger.error(f"Target coordinates: x={x_center}, y={y_center}, w={width}, h={height}")
            logger.error("Available detections:")
            for line in lines:
                logger.error(f"  {line.strip()}")
            raise HTTPException(
                status_code=404,
                detail="Detection not found in label file. Coordinates don't match any existing detection."
            )
       
        # Write back the updated labels
        with open(label_file, 'w') as f:
            f.writelines(new_lines)
       
        # Save corrected label file
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        corrected_label_filename = f"corrected_{timestamp}_{os.path.basename(label_file)}"
        corrected_label_path = os.path.join(CORRECTED_LABELS_DIR, corrected_label_filename)
       
        with open(corrected_label_path, 'w') as f:
            f.writelines(new_lines)
       
        logger.info(f"Saved corrected labels to: {corrected_label_path}")
       
        # Find and load the original image
        original_image_path = None
        for fname in os.listdir(STORAGE_BASE_DIR):
            if data['fileName'] in fname:
                original_image_path = os.path.join(STORAGE_BASE_DIR, fname)
                break
       
        if not original_image_path:
            raise HTTPException(status_code=404, detail="Original image not found")
       
        # Load and process the image
        img = Image.open(original_image_path).convert("RGB")
        img_array = np.array(img)
       
        # Create YOLO format boxes for remaining detections
        boxes = []
        confidences = []
        class_ids = []
       
        img_width, img_height = img.size
        for det in remaining_detections:
            # Convert normalized coordinates to pixel coordinates
            x_center = det['x_center'] * img_width
            y_center = det['y_center'] * img_height
            width = det['width'] * img_width
            height = det['height'] * img_height
           
            # Convert to xyxy format for YOLO
            x1 = x_center - width/2
            y1 = y_center - height/2
            x2 = x_center + width/2
            y2 = y_center + height/2
           
            boxes.append([x1, y1, x2, y2])
            confidences.append(0.9)  # Use high confidence for existing detections
            class_ids.append(det['class_id'])
       
        # Convert to numpy arrays
        boxes = np.array(boxes)
        confidences = np.array(confidences)
        class_ids = np.array(class_ids)
       
        # Create a Results object for YOLO visualization
        results = Results(
            orig_img=img_array,
            path=original_image_path,
            names=model.names,
            boxes=None
        )
        if len(boxes) > 0:
            # Convert boxes to tensor format expected by YOLO
            boxes_tensor = np.concatenate([
                boxes,  # xyxy
                confidences.reshape(-1, 1),  # Use actual confidence values
                class_ids.reshape(-1, 1)  # class ids
            ], axis=1)
           
            # Create Boxes object with tensor and original shape
            results.boxes = Boxes(boxes_tensor, img_array.shape)
       
        # Use YOLO's visualization with custom settings
        annotated_img = results.plot(labels=True, conf=False)  # Hide confidence in labels
        annotated_pil = Image.fromarray(annotated_img)
       
        # Save the corrected image with a distinct filename
        corrected_filename = f"corrected_{timestamp}_{data['fileName']}"
        corrected_path = os.path.join(CORRECTED_DIR, corrected_filename)
        annotated_pil.save(corrected_path, format='JPEG', quality=95)
       
        logger.info(f"Saved corrected image to: {corrected_path}")
       
        # Convert to base64 for frontend
        img_buffer = io.BytesIO()
        annotated_pil.save(img_buffer, format='JPEG', quality=95)
        img_base64 = base64.b64encode(img_buffer.getvalue()).decode()
       
        # Convert remaining detections to frontend format
        frontend_detections = []
        for det in remaining_detections:
            x_center = det['x_center'] * img_width
            y_center = det['y_center'] * img_height
            width = det['width'] * img_width
            height = det['height'] * img_height
           
            x = x_center - width/2
            y = y_center - height/2
           
            frontend_detections.append({
                "className": model.names[det['class_id']],
                "confidence": det.get('confidence', 0.9),  # Use original confidence or default to 0.9
                "x": int(x),
                "y": int(y),
                "width": int(width),
                "height": int(height)
            })
       
        # Log the deletion
        log_entry = {
            "timestamp": datetime.now().isoformat(),
            "filename": filename,
            "action": "detection_delete",
            "deleted_class": data.get('className'),
            "coordinates": {
                "x_center": x_center,
                "y_center": y_center,
                "width": width,
                "height": height
            }
        }
       
        with open(os.path.join(LOGS_DIR, "label_updates.log"), "a") as f:
            f.write(json.dumps(log_entry) + "\n")
       
        logger.info(f"Successfully deleted detection from {filename}")
       
        return {
            "message": "Detection deleted successfully",
            "details": {
                "filename": filename,
                "deleted_class": data.get('className'),
                "normalized_coords": {
                    "x_center": x_center,
                    "y_center": y_center,
                    "width": width,
                    "height": height
                }
            },
            "updatedImage": f"data:image/jpeg;base64,{img_base64}",
            "remainingDetections": frontend_detections,
            "correctedImagePath": corrected_path,
            "correctedLabelPath": corrected_label_path,
            "downloadUrl": f"/api/download/corrected/{os.path.basename(corrected_path)}"
        }
       
    except HTTPException as he:
        logger.error(f"HTTP error in delete_detection: {he.detail}")
        raise he
    except Exception as e:
        logger.error(f"Unexpected error in delete_detection: {str(e)}")
        logger.exception(e)
        raise HTTPException(
            status_code=500,
            detail=f"Server error while deleting detection: {str(e)}"
        )


# Add new endpoint to download corrected images
@app.get("/api/download/corrected/{filename}")
async def download_corrected_image(filename: str):
    """Download a corrected image with updated bounding boxes"""
    file_path = os.path.join(CORRECTED_DIR, filename)
   
    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="Corrected image not found")
   
    logger.info(f"Downloading corrected image: {filename}")
    return FileResponse(
        path=file_path,
        filename=filename,
        media_type='image/jpeg'
    )


@app.post("/api/batch-save-changes")
async def batch_save_changes(data: dict):
    """Save multiple changes (deletions/edits) to an image and its labels at once"""
    try:
        filename = data.get('fileName')
        changes = data.get('changes', [])  # List of changes made
        original_confidences = data.get('originalConfidences', {})  # Map of detection index to original confidence
       
        if not filename or not changes:
            raise HTTPException(status_code=400, detail="Missing filename or changes")
           
        logger.info(f"Processing batch changes for {filename}")
        logger.info(f"Number of changes: {len(changes)}")
       
        # Find the corresponding label file
        label_file = find_label_file(filename)
        if not label_file:
            raise HTTPException(status_code=404, detail=f"Label file not found for {filename}")
       
        # Read the current labels
        with open(label_file, 'r') as f:
            lines = f.readlines()
           
        # Process all changes
        new_lines = lines.copy()
        remaining_detections = []
        deleted_indices = set()  # Keep track of deleted detection indices
       
        # First pass: Process deletions
        for change in changes:
            if change['type'] == 'deletion':
                coords = change.get('coordinates', {})
                x_center = coords.get('x_center')
                y_center = coords.get('y_center')
                width = coords.get('width')
                height = coords.get('height')
               
                # Find and mark the detection for deletion
                for i, line in enumerate(lines):
                    if i in deleted_indices:
                        continue
                       
                    parts = line.strip().split()
                    if len(parts) == 5:
                        current_x = float(parts[1])
                        current_y = float(parts[2])
                        current_w = float(parts[3])
                        current_h = float(parts[4])
                       
                        tolerance = 0.02
                        if (abs(current_x - x_center) < tolerance and
                            abs(current_y - y_center) < tolerance and
                            abs(current_w - width) < tolerance and
                            abs(current_h - height) < tolerance):
                            deleted_indices.add(i)
                            break
       
        # Second pass: Process edits and build remaining detections
        for i, line in enumerate(lines):
            if i in deleted_indices:
                continue
               
            parts = line.strip().split()
            if len(parts) != 5:
                continue
               
            # Check if this detection was edited
            detection_edited = False
            for change in changes:
                if change['type'] == 'edit':
                    coords = change.get('coordinates', {})
                    current_x = float(parts[1])
                    current_y = float(parts[2])
                    current_w = float(parts[3])
                    current_h = float(parts[4])
                   
                    tolerance = 0.02
                    if (abs(current_x - coords.get('x_center', 0)) < tolerance and
                        abs(current_y - coords.get('y_center', 0)) < tolerance and
                        abs(current_w - coords.get('width', 0)) < tolerance and
                        abs(current_h - coords.get('height', 0)) < tolerance):
                        # Update class for this detection
                        class_id = get_class_id(change['newClass'])
                        new_lines[i] = f"{class_id} {parts[1]} {parts[2]} {parts[3]} {parts[4]}\n"
                        detection_edited = True
                        break
           
            # Add to remaining detections
            if not detection_edited:
                remaining_detections.append({
                    "class_id": int(parts[0]),
                    "x_center": float(parts[1]),
                    "y_center": float(parts[2]),
                    "width": float(parts[3]),
                    "height": float(parts[4]),
                    "confidence": original_confidences.get(str(i), 0.9)  # Use original confidence or default
                })
       
        # Save corrected label file
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        corrected_label_filename = f"corrected_{timestamp}_{os.path.basename(label_file)}"
        corrected_label_path = os.path.join(CORRECTED_LABELS_DIR, corrected_label_filename)
       
        with open(corrected_label_path, 'w') as f:
            f.writelines(new_lines)
       
        logger.info(f"Saved corrected labels to: {corrected_label_path}")
       
        # Find and load the original image
        original_image_path = None
        for fname in os.listdir(STORAGE_BASE_DIR):
            if filename in fname:
                original_image_path = os.path.join(STORAGE_BASE_DIR, fname)
                break
       
        if not original_image_path:
            raise HTTPException(status_code=404, detail="Original image not found")
       
        # Load and process the image
        img = Image.open(original_image_path).convert("RGB")
        img_array = np.array(img)
       
        # Create YOLO format boxes for remaining detections
        boxes = []
        confidences = []
        class_ids = []
       
        img_width, img_height = img.size
        for det in remaining_detections:
            # Convert normalized coordinates to pixel coordinates
            x_center = det['x_center'] * img_width
            y_center = det['y_center'] * img_height
            width = det['width'] * img_width
            height = det['height'] * img_height
           
            # Convert to xyxy format for YOLO
            x1 = x_center - width/2
            y1 = y_center - height/2
            x2 = x_center + width/2
            y2 = y_center + height/2
           
            boxes.append([x1, y1, x2, y2])
            confidences.append(det['confidence'])  # Use the original confidence
            class_ids.append(det['class_id'])
       
        # Convert to numpy arrays
        boxes = np.array(boxes) if boxes else np.zeros((0, 4))
        confidences = np.array(confidences) if confidences else np.zeros(0)
        class_ids = np.array(class_ids) if class_ids else np.zeros(0)
       
        # Create a Results object for YOLO visualization
        results = Results(
            orig_img=img_array,
            path=original_image_path,
            names=model.names,
            boxes=None
        )
        if len(boxes) > 0:
            # Convert boxes to tensor format expected by YOLO
            boxes_tensor = np.concatenate([
                boxes,  # xyxy
                confidences.reshape(-1, 1),  # Use actual confidence values
                class_ids.reshape(-1, 1)  # class ids
            ], axis=1)
           
            # Create Boxes object with tensor and original shape
            results.boxes = Boxes(boxes_tensor, img_array.shape)
       
        # Use YOLO's visualization
        annotated_img = results.plot()
        annotated_pil = Image.fromarray(annotated_img)
       
        # Save the corrected image
        corrected_filename = f"corrected_{timestamp}_{filename}"
        corrected_path = os.path.join(CORRECTED_DIR, corrected_filename)
        annotated_pil.save(corrected_path, format='JPEG', quality=95)
       
        logger.info(f"Saved corrected image to: {corrected_path}")
       
        # Convert to base64 for frontend
        img_buffer = io.BytesIO()
        annotated_pil.save(img_buffer, format='JPEG', quality=95)
        img_base64 = base64.b64encode(img_buffer.getvalue()).decode()
       
        # Convert remaining detections to frontend format
        frontend_detections = []
        for det in remaining_detections:
            x_center = det['x_center'] * img_width
            y_center = det['y_center'] * img_height
            width = det['width'] * img_width
            height = det['height'] * img_height
           
            x = x_center - width/2
            y = y_center - height/2
           
            frontend_detections.append({
                "className": model.names[det['class_id']],
                "confidence": det['confidence'],
                "x": int(x),
                "y": int(y),
                "width": int(width),
                "height": int(height)
            })
       
        return {
            "message": "Changes saved successfully",
            "details": {
                "filename": filename,
                "changesApplied": len(changes)
            },
            "updatedImage": f"data:image/jpeg;base64,{img_base64}",
            "remainingDetections": frontend_detections,
            "correctedImagePath": corrected_path,
            "correctedLabelPath": corrected_label_path,
            "downloadUrl": f"/api/download/corrected/{os.path.basename(corrected_path)}"
        }
       
    except Exception as e:
        logger.error(f"Error in batch_save_changes: {str(e)}")
        logger.exception(e)
        raise HTTPException(
            status_code=500,
            detail=f"Server error while saving changes: {str(e)}"
        )


# Add at the top with other global variables
TRAINING_STATUS = {
    "is_training": False,
    "start_time": None,
    "progress": None
}

@app.post("/api/retrain")
async def retrain_model():
    """Retrain the model using corrected labels"""
    global TRAINING_STATUS
    
    try:
        if TRAINING_STATUS["is_training"]:
            raise HTTPException(
                status_code=400,
                detail="Training is already in progress"
            )
            
        logger.info("Starting model retraining process")
        TRAINING_STATUS["is_training"] = True
        TRAINING_STATUS["start_time"] = datetime.now().isoformat()
        
        # Create a directory for the new model
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        new_model_dir = os.path.join(MODELS_DIR, f"model_{timestamp}")
        os.makedirs(new_model_dir, exist_ok=True)
        
        # Create dataset.yaml for training
        dataset_yaml = {
            'path': os.path.abspath(os.path.join(os.getcwd())),  # Root directory
            'train': os.path.abspath(CORRECTED_DIR),  # Training images
            'val': os.path.abspath(CORRECTED_DIR),    # Validation images (using same for demo)
            'names': model.names  # Keep the same class names
        }
        
        yaml_path = os.path.join(new_model_dir, 'dataset.yaml')
        with open(yaml_path, 'w') as f:
            import yaml
            yaml.dump(dataset_yaml, f)
        
        # Start training in a background task
        training_task = asyncio.create_task(run_training(new_model_dir, yaml_path))
        
        return {
            "message": "Training started successfully",
            "model_dir": new_model_dir,
            "status": "training"
        }
        
    except Exception as e:
        TRAINING_STATUS["is_training"] = False
        TRAINING_STATUS["progress"] = None
        logger.error(f"Error starting retraining: {str(e)}")
        logger.exception(e)
        raise HTTPException(
            status_code=500,
            detail=f"Error starting retraining: {str(e)}"
        )

async def run_training(model_dir: str, yaml_path: str):
    """Run the actual training process"""
    global model, TRAINING_STATUS
    try:
        # Get the current model as starting point
        current_model_path = model.ckpt_path
        
        # Force CPU usage and optimize settings
        device = 'cpu'
        logger.info("Using CPU for training")
        
        # Training arguments
        args = dict(
            data=yaml_path,           # Path to data config file
            model=current_model_path, # Path to model file
            epochs=10,               # Reduced number of epochs
            imgsz=640,              # Image size
            batch=4,                # Small batch size for CPU
            device=device,          # Force CPU
            project=model_dir,      # Save results to project/name
            name='exp',             # Save results to project/name
            exist_ok=True,          # Existing project/name ok
            pretrained=True,        # Use pretrained model
            optimizer='Adam',       # Optimizer
            verbose=True,           # Verbose output
            seed=42,               # Random seed
            deterministic=True,     # Deterministic mode
            box=7.5,               # Box loss gain
            cls=0.5,               # Cls loss gain
            dfl=1.5,               # DFL loss gain
            plots=True,            # Save plots
            save=True,             # Save training results
            workers=0              # No worker processes for CPU training
        )
        
        # Start training
        logger.info(f"Starting training with args: {args}")
        TRAINING_STATUS["progress"] = "Initializing training on CPU (Quick training mode)..."
        
        results = await asyncio.to_thread(lambda: YOLO(current_model_path).train(**args))
        
        # Save training results
        with open(os.path.join(model_dir, 'training_results.json'), 'w') as f:
            json.dump(results, f, indent=2)
            
        # Calculate and save training summary
        results_file = os.path.join(model_dir, 'exp', 'results.csv')
        if os.path.exists(results_file):
            df = pd.read_csv(results_file)
            
            # Calculate metrics summary
            summary = {
                'initial_cls_loss': float(df['train/cls_loss'].iloc[0]),
                'final_cls_loss': float(df['train/cls_loss'].iloc[-1]),
                'cls_loss_reduction': float(df['train/cls_loss'].iloc[0] - df['train/cls_loss'].iloc[-1]),
                'initial_box_loss': float(df['train/box_loss'].iloc[0]) if 'train/box_loss' in df else 0,
                'final_box_loss': float(df['train/box_loss'].iloc[-1]) if 'train/box_loss' in df else 0,
                'box_loss_reduction': float(df['train/box_loss'].iloc[0] - df['train/box_loss'].iloc[-1]) if 'train/box_loss' in df else 0,
                'epochs_completed': len(df),
                'final_precision': float(df['metrics/precision(B)'].iloc[-1]) if 'metrics/precision(B)' in df else 0,
                'final_recall': float(df['metrics/recall(B)'].iloc[-1]) if 'metrics/recall(B)' in df else 0,
                'final_mAP50': float(df['metrics/mAP50(B)'].iloc[-1]) if 'metrics/mAP50(B)' in df else 0
            }
            
            # Save summary
            with open(os.path.join(model_dir, 'training_summary.json'), 'w') as f:
                json.dump(summary, f, indent=2)
            
            # Create and save training plots
            plt.figure(figsize=(12, 8))
            
            # Plot losses
            plt.subplot(2, 1, 1)
            plt.plot(df['epoch'], df['train/cls_loss'], label='Classification Loss')
            if 'train/box_loss' in df.columns:
                plt.plot(df['epoch'], df['train/box_loss'], label='Box Loss')
            plt.title('Training Losses')
            plt.xlabel('Epoch')
            plt.ylabel('Loss')
            plt.legend()
            plt.grid(True)
            
            # Plot metrics
            plt.subplot(2, 1, 2)
            metrics = ['metrics/precision(B)', 'metrics/recall(B)', 'metrics/mAP50(B)']
            for metric in metrics:
                if metric in df.columns:
                    plt.plot(df['epoch'], df[metric], label=metric.split('/')[1])
            plt.title('Training Metrics')
            plt.xlabel('Epoch')
            plt.ylabel('Value')
            plt.legend()
            plt.grid(True)
            
            plt.tight_layout()
            plt.savefig(os.path.join(model_dir, 'training_metrics.png'))
            plt.close()
            
            # Update training status with summary
            progress_text = f"""Quick Training completed ({len(df)} epochs):
            - Classification loss reduced by {summary['cls_loss_reduction']:.4f} (from {summary['initial_cls_loss']:.4f} to {summary['final_cls_loss']:.4f})
            - Box loss reduced by {summary['box_loss_reduction']:.4f} (from {summary['initial_box_loss']:.4f} to {summary['final_box_loss']:.4f})
            - Final metrics: Precision={summary['final_precision']:.4f}, Recall={summary['final_recall']:.4f}, mAP50={summary['final_mAP50']:.4f}
            - Training plots saved to {os.path.join(model_dir, 'training_metrics.png')}"""
            
            TRAINING_STATUS["progress"] = progress_text
        
        logger.info("Training completed successfully")
        
        # Update the model to use the new weights
        best_weights = os.path.join(model_dir, 'exp', 'weights', 'best.pt')
        if os.path.exists(best_weights):
            model = YOLO(best_weights)
            logger.info(f"Model updated to use new weights: {best_weights}")
            
        TRAINING_STATUS["is_training"] = False
        
    except Exception as e:
        logger.error(f"Error during training: {str(e)}")
        logger.exception(e)
        TRAINING_STATUS["is_training"] = False
        TRAINING_STATUS["progress"] = f"failed: {str(e)}"

@app.get("/api/training/status")
async def get_training_status():
    """Get the current training status"""
    try:
        if not TRAINING_STATUS["is_training"]:
            if TRAINING_STATUS["progress"] == "completed":
                return {"status": "completed"}
            return {"status": "idle"}
            
        return {
            "status": "training",
            "start_time": TRAINING_STATUS["start_time"],
            "progress": TRAINING_STATUS["progress"]
        }
        
    except Exception as e:
        logger.error(f"Error getting training status: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Error getting training status: {str(e)}"
        )

@app.get("/api/training/metrics")
async def get_training_metrics():
    """Get training metrics visualization"""
    try:
        # Find the most recent training directory
        model_dirs = [d for d in Path(MODELS_DIR).glob("model_*") if d.is_dir()]
        if not model_dirs:
            raise HTTPException(status_code=404, detail="No training data found")
            
        latest_model_dir = max(model_dirs, key=lambda x: x.stat().st_mtime)
        results_file = latest_model_dir / "exp" / "results.csv"
        
        if not results_file.exists():
            raise HTTPException(status_code=404, detail="No training results found")
            
        # Read and process the results
        df = pd.read_csv(results_file)
        
        # Create visualization
        plt.figure(figsize=(12, 8))
        
        # Plot training losses
        plt.subplot(2, 1, 1)
        plt.plot(df['epoch'], df['train/cls_loss'], label='Classification Loss')
        if 'train/box_loss' in df.columns:
            plt.plot(df['epoch'], df['train/box_loss'], label='Box Loss')
        plt.title('Training Losses')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True)
        
        # Plot metrics if available
        plt.subplot(2, 1, 2)
        metrics = ['metrics/precision(B)', 'metrics/recall(B)', 'metrics/mAP50(B)']
        for metric in metrics:
            if metric in df.columns:
                plt.plot(df['epoch'], df[metric], label=metric.split('/')[1])
        plt.title('Training Metrics')
        plt.xlabel('Epoch')
        plt.ylabel('Value')
        plt.legend()
        plt.grid(True)
        
        plt.tight_layout()
        
        # Save plot to bytes
        buf = io.BytesIO()
        plt.savefig(buf, format='png')
        buf.seek(0)
        plt.close()
        
        return StreamingResponse(buf, media_type="image/png")
        
    except Exception as e:
        logger.error(f"Error generating training metrics: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/training/batch-examples")
async def get_training_batch_examples():
    """Get training batch examples"""
    try:
        # Find the most recent training directory
        model_dirs = [d for d in Path(MODELS_DIR).glob("model_*") if d.is_dir()]
        if not model_dirs:
            raise HTTPException(status_code=404, detail="No training data found")
            
        latest_model_dir = max(model_dirs, key=lambda x: x.stat().st_mtime)
        batch_images = list((latest_model_dir / "exp").glob("train_batch*.jpg"))
        
        if not batch_images:
            raise HTTPException(status_code=404, detail="No training batch images found")
            
        # Return the latest batch image
        latest_batch = max(batch_images, key=lambda x: x.stat().st_mtime)
        return FileResponse(latest_batch)
        
    except Exception as e:
        logger.error(f"Error getting batch examples: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

