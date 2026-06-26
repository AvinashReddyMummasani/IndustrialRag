# src/parsers/archive_parser.py
import zipfile
import tarfile
import uuid
from pathlib import Path
from typing import List

class ArchiveProcessor:
    """Safely extracts container formats and yields individual file paths."""
    
    def __init__(self, extract_dir: Path, max_files: int = 1000, max_size_bytes: int = 5 * 1024 * 1024 * 1024): # 5GB limit
        self.extract_dir = extract_dir
        self.max_files = max_files
        self.max_size_bytes = max_size_bytes
    
    def can_handle(self, file_path: Path) -> bool:
        """
        Determines if the file is a compressed archive.
        """
        valid_extensions = {'.zip', '.tar', '.gz', '.rar', '.7z'}
        # Note: A file like archive.tar.gz will yield '.gz' as the suffix, which is caught here.
        return file_path.suffix.lower() in valid_extensions

    def unpack(self, file_path: Path, archive_id: str) -> List[Path]:
        """Extracts the archive and returns a list of paths to the extracted files."""
        target_dir = self.extract_dir / f"unpacked_{archive_id}"
        target_dir.mkdir(parents=True, exist_ok=True)
        extracted_paths = []
        total_size = 0
        file_count = 0

        try:
            if zipfile.is_zipfile(file_path):
                with zipfile.ZipFile(file_path, 'r') as zf:
                    for info in zf.infolist():
                        if info.is_dir(): continue
                        file_count += 1
                        total_size += info.file_size
                        
                        self._check_limits(file_count, total_size, file_path.name)
                        
                        extracted_path = Path(zf.extract(info, target_dir))
                        extracted_paths.append(extracted_path)
                        
            elif tarfile.is_tarfile(file_path):
                with tarfile.open(file_path, 'r:*') as tf:
                    for info in tf.getmembers():
                        if info.isdir(): continue
                        file_count += 1
                        total_size += info.size
                        
                        self._check_limits(file_count, total_size, file_path.name)
                        
                        tf.extract(info, target_dir)
                        extracted_paths.append(target_dir / info.name)
            else:
                raise ValueError(f"Unsupported archive format: {file_path}")
                
        except Exception as e:
            # If extraction fails midway (e.g. limit hit), clean up the partial extraction
            import shutil
            shutil.rmtree(target_dir, ignore_errors=True)
            raise RuntimeError(f"Archive extraction failed: {str(e)}")

        return extracted_paths

    def _check_limits(self, count: int, size: int, name: str):
        if count > self.max_files:
            raise SecurityError(f"Archive {name} exceeded max file limit ({self.max_files}). Possible zip bomb.")
        if size > self.max_size_bytes:
            raise SecurityError(f"Archive {name} exceeded max size limit. Possible zip bomb.")

class SecurityError(Exception):
    pass