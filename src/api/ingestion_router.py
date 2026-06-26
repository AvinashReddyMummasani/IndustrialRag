from fastapi import APIRouter, UploadFile, File, HTTPException, BackgroundTasks, Request
import uuid
from pathlib import Path
import logging

logger = logging.getLogger(__name__)

router = APIRouter()
UPLOAD_DIR = Path("./temp_uploads")
UPLOAD_DIR.mkdir(exist_ok=True)
ALLOWED_EXTENSIONS = {".pdf", ".png", ".jpg", ".jpeg", ".csv", ".xlsx", ".eml"}
MAX_FILE_SIZE_MB = 50 

def cleanup_temp_file(path: Path):
    """Fallback cleanup if the background task fails."""
    if path.exists():
        try:
            path.unlink()
        except OSError as e:
            logger.error(f"Failed to delete temp file {path}: {e}")

@router.post("/upload", status_code=202)
async def upload_document(
    request: Request,
    background_tasks: BackgroundTasks, 
    file: UploadFile = File(...)
):
    # 1. Extract pipeline from the application state (Memory Safe)
    if not hasattr(request.app.state, "pipeline") or request.app.state.pipeline is None:
        logger.error("IngestionPipeline not found in application state.")
        raise HTTPException(status_code=503, detail="The ingestion pipeline is currently offline.")
        
    pipeline = request.app.state.pipeline

    # 2. File Validation
    file_ext = Path(file.filename).suffix.lower()
    if file_ext not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=400, 
            detail=f"Invalid extension. Allowed: {', '.join(ALLOWED_EXTENSIONS)}"
        )

    file_id = str(uuid.uuid4())
    temp_file_path = UPLOAD_DIR / f"{file_id}{file_ext}"

    # 3. Disk I/O with Size Constraints
    try:
        size = 0
        with open(temp_file_path, "wb") as buffer:
            while chunk := await file.read(1024 * 1024): # 1MB chunks
                size += len(chunk)
                if size > MAX_FILE_SIZE_MB * 1024 * 1024:
                    buffer.close()
                    cleanup_temp_file(temp_file_path)
                    raise HTTPException(status_code=413, detail="File exceeds maximum allowed size.")
                buffer.write(chunk)
                
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed disk write for {file.filename}: {e}")
        cleanup_temp_file(temp_file_path)
        raise HTTPException(status_code=500, detail="Failed to persist file to disk.")

    # 4. Offload heavy processing to worker pool
    background_tasks.add_task(
        pipeline.process_file, 
        temp_file_path, 
        file_id, 
        file.filename
    )
    
    return {
        "status": "processing", 
        "document_id": file_id,
        "message": "File accepted and added to the ingestion queue."
    }