from fastapi import FastAPI, File, UploadFile
from fastapi.responses import FileResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import shutil
import os
import subprocess
import uuid

app = FastAPI()

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],  # Your Next.js app's URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

CHECKPOINT_PATH = "checkpoints/model.pth"
CONFIG_PATH = "config/wireframe.yaml"
UPLOAD_DIR = "uploads"
RESULT_DIR = "results"

os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(RESULT_DIR, exist_ok=True)

@app.post("/process/")
async def process_image(file: UploadFile = File(...)):
    # Save uploaded file
    file_id = str(uuid.uuid4())
    input_path = os.path.join(UPLOAD_DIR, f"{file_id}_{file.filename}")
    with open(input_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    # Prepare output path (modify as needed)
    output_path = os.path.join(RESULT_DIR, f"{file_id}_output.svg")

    # Run your model (adjust command as needed)
    cmd = [
        "python", "run.py", "-d", "0", CONFIG_PATH, CHECKPOINT_PATH, input_path, output_path
    ]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
    except subprocess.CalledProcessError as e:
        return JSONResponse(status_code=500, content={"error": e.stderr, "stdout": e.stdout})

    # Check for output file
    if os.path.exists(output_path):
        return FileResponse(output_path, media_type="image/svg+xml")
    else:
        # Return subprocess output for debugging
        return JSONResponse(status_code=500, content={
            "error": "Processing failed",
            "stdout": result.stdout,
            "stderr": result.stderr
        })

@app.get("/")
def read_root():
    return {"message": "Wireframe API is running."}