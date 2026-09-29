"""File management service: downloads, cleanup, and storage tracking."""

import hashlib
import io
import json
import logging
import os
import shutil
import tempfile
import uuid
import zipfile
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Optional, Dict, Tuple

logger = logging.getLogger(__name__)


class FileManager:
    """Manages image files, downloads, and cleanup."""

    def __init__(self, uploads_dir: Path, db_manager):
        self.uploads_dir = Path(uploads_dir)
        self.db_manager = db_manager
        self.uploads_dir.mkdir(exist_ok=True)

    @staticmethod
    def hash_bytes(data: bytes) -> str:
        """SHA-256 hash from in-memory bytes — no disk I/O."""
        sha256 = hashlib.sha256(data)
        return sha256.hexdigest()

    def calculate_hash(self, file_path: str) -> str:
        """Calculate SHA256 hash of file from disk."""
        sha256 = hashlib.sha256()
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                sha256.update(chunk)
        return sha256.hexdigest()

    def get_file_size(self, file_path: str) -> int:
        """Get file size in bytes."""
        return os.path.getsize(file_path)

    def update_image_with_file_info(
        self,
        image_id: int,
        file_path: str,
        confidence: float,
        detected_animal: str = "",
        precomputed_hash: Optional[str] = None,
    ):
        """Store file metadata (hash, size, tier). Uses precomputed_hash when provided."""
        try:
            file_hash = precomputed_hash or self.calculate_hash(file_path)
            file_size = self.get_file_size(file_path)
            animal = (detected_animal or "").strip()
            is_person_or_vehicle = animal and (
                animal.lower() in ("person", "vehicle", "human")
                or "person" in animal.lower()
                or "vehicle" in animal.lower()
            )
            if is_person_or_vehicle or not animal or confidence == 0:
                has_animal = False
                file_tier = "empty"
            elif confidence > 0.4:
                has_animal = True
                file_tier = "valid"
            else:
                has_animal = True
                file_tier = "low_conf"

            self.db_manager.update_image_file_info(
                image_id, file_hash, file_size, has_animal, file_tier
            )
            return {"hash": file_hash, "size": file_size, "tier": file_tier}
        except Exception as e:
            logger.error(f"Error updating file info for image {image_id}: {e}")
            raise

    def reconcile_missing_files(self) -> int:
        """Mark DB records 'missing' when the file no longer exists on disk.

        Runs once at startup to clean up stale records left by container
        rebuilds or manual file removal.  Returns the count of newly-marked rows.
        """
        rows = self.db_manager.get_available_image_filenames()
        scrubbed_dir = self.uploads_dir / "scrubbed"
        missing_ids = []
        for row in rows:
            image_id, filename = row[0], row[1]
            original = self.uploads_dir / filename
            scrubbed = scrubbed_dir / Path(filename).name
            if not original.exists() and not scrubbed.exists():
                missing_ids.append(image_id)
        if missing_ids:
            marked = self.db_manager.mark_files_missing(missing_ids)
            logger.info("Reconciliation: marked %d image(s) as missing (file not on disk)", marked)
            return marked
        return 0

    # ──────────────────────────────────────────────────────────────────
    # Download Operations
    # ──────────────────────────────────────────────────────────────────

    def create_batch_download(
        self,
        image_ids: List[int],
        download_type: str = "batch",
        include_metadata: bool = True,
    ) -> Dict:
        """Create a batch download package (ZIP)."""
        download_id = str(uuid.uuid4())[:8]

        try:
            total_size = sum(
                img[2] for img in self.db_manager.get_images_by_tier("", limit=10000)
                if img[0] in image_ids
            )

            self.db_manager.create_download(download_id, download_type, image_ids, total_size)

            return {
                "download_id": download_id,
                "status": "preparing",
                "image_count": len(image_ids),
                "total_size_mb": total_size / (1024 * 1024),
                "expires_at": (datetime.utcnow() + timedelta(days=1)).isoformat(),
            }
        except Exception as e:
            logger.error(f"Error creating batch download: {e}")
            raise

    def get_batch_zip_stream(
        self,
        download_id: str,
        image_ids: List[int],
        include_metadata: bool = True,
    ) -> Tuple[io.BytesIO, str]:
        """Generate ZIP file stream for batch download."""
        zip_buffer = io.BytesIO()
        file_hash = hashlib.sha256()
        added_count = 0

        try:
            with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as zf:
                # One physical file may have multiple image_ids (one per detection
                # crop). Track seen filenames so each file is written only once.
                seen_filenames: set = set()
                for image_id in image_ids:
                    try:
                        conn = self.db_manager.get_connection()
                        cursor = conn.cursor()
                        cursor.execute(
                            "SELECT filename FROM images WHERE id = ?", (image_id,)
                        )
                        result = cursor.fetchone()
                        conn.close()

                        if not result:
                            logger.debug(f"Image {image_id} not found in DB")
                            continue

                        filename = result[0]
                        if filename in seen_filenames:
                            continue
                        seen_filenames.add(filename)

                        file_path = self.uploads_dir / filename

                        if file_path.exists():
                            with open(file_path, "rb") as f:
                                file_data = f.read()
                                file_hash.update(file_data)
                                zf.writestr(f"images/{filename}", file_data)
                                added_count += 1
                        else:
                            logger.warning(f"File not found: {file_path}")
                    except Exception as e:
                        logger.warning(f"Could not add image {image_id} to ZIP: {e}")
                        continue

                # Add metadata file if requested.
                # Uses seen_filenames (the deduplicated set from the loop above)
                # so the metadata rows match exactly the files in the ZIP.
                if include_metadata and seen_filenames:
                    try:
                        conn = self.db_manager.get_connection()
                        cursor = conn.cursor()

                        # Fetch one image record per filename (most recent id)
                        # plus all detection rows for that image.
                        placeholders = ",".join("?" * len(seen_filenames))
                        cursor.execute(
                            f"""
                            SELECT
                                i.id, i.filename, i.station_id,
                                i.capture_date, i.capture_time, i.temperature,
                                i.day_night, i.brightness, i.file_size_bytes,
                                i.file_hash, i.file_tier, i.user_notes,
                                i.processed_at,
                                d.detected_animal, d.confidence,
                                d.speciesnet_confidence, d.method, d.bbox, d.ide_id
                            FROM images i
                            LEFT JOIN detections d ON d.image_id = i.id
                            WHERE i.filename IN ({placeholders})
                              AND i.id IN (
                                  SELECT MAX(id) FROM images
                                  WHERE filename IN ({placeholders})
                                  GROUP BY filename
                              )
                            ORDER BY i.filename, d.confidence DESC
                            """,
                            list(seen_filenames) * 2,
                        )
                        rows = cursor.fetchall()
                        conn.close()

                        # Group detections under their parent image
                        images_map: dict = {}
                        for row in rows:
                            fname = row[1]
                            if fname not in images_map:
                                images_map[fname] = {
                                    "filename":        fname,
                                    "station_id":      row[2],
                                    "capture_date":    row[3],
                                    "capture_time":    row[4],
                                    "temperature":     row[5],
                                    "day_night":       row[6],
                                    "brightness":      row[7],
                                    "file_size_bytes": row[8],
                                    "file_hash":       row[9],
                                    "tier":            row[10],
                                    "user_notes":      row[11] or "",
                                    "processed_at":    row[12],
                                    "detections":      [],
                                }
                            if row[13]:  # detected_animal
                                images_map[fname]["detections"].append({
                                    "species":               row[13],
                                    "confidence":            row[14],
                                    "speciesnet_confidence": row[15],
                                    "method":                row[16],
                                    "bbox":                  json.loads(row[17]) if row[17] else None,
                                    "ide_id":                row[18],
                                })

                        metadata = sorted(images_map.values(), key=lambda x: (x["capture_date"] or "", x["capture_time"] or ""))
                        metadata_json = json.dumps(metadata, indent=2)
                        zf.writestr("metadata.json", metadata_json)
                        file_hash.update(metadata_json.encode())
                    except Exception as e:
                        logger.warning(f"Could not add metadata: {e}")

                # Add README
                readme = f"""ViumbeLens Download Package
Generated: {datetime.utcnow().isoformat()}
Total images: {added_count}
Download ID: {download_id}

Contents:
- images/: Original uploaded image files
- metadata.json: Image metadata (capture date, tier, etc.)
- README.txt: This file
"""
                zf.writestr("README.txt", readme)

            zip_buffer.seek(0)
            hash_hex = file_hash.hexdigest()
            self.db_manager.complete_download(download_id, hash_hex)
            logger.info(f"ZIP created for download {download_id}: {added_count} images")

            return zip_buffer, hash_hex

        except Exception as e:
            logger.error(f"Error creating ZIP stream: {e}")
            raise

    # ──────────────────────────────────────────────────────────────────
    # File artifact deletion
    # ──────────────────────────────────────────────────────────────────

    def delete_image_files(self, filename: str, image_id: Optional[int] = None) -> int:
        """Delete all on-disk artifacts for one image. Returns bytes freed."""
        safe_name = Path(filename or "upload").name
        paths: List[Path] = []

        original = self.uploads_dir / safe_name
        if original.is_file():
            paths.append(original)

        scrubbed = self.uploads_dir / "scrubbed" / safe_name
        if scrubbed.is_file():
            paths.append(scrubbed)

        thumbs_dir = self.uploads_dir / "thumbs"
        if thumbs_dir.is_dir():
            thumb_names = {f"{safe_name}.jpg", f"scrub_{safe_name}.jpg"}
            if image_id is not None:
                thumb_names.add(f"{image_id}_{safe_name}.jpg")
                thumb_names.add(f"{image_id}_scrub_{safe_name}.jpg")
            for w_dir in thumbs_dir.iterdir():
                if not w_dir.is_dir():
                    continue
                for name in thumb_names:
                    candidate = w_dir / name
                    if candidate.is_file():
                        paths.append(candidate)

        freed = 0
        for path in paths:
            try:
                freed += path.stat().st_size
                path.unlink()
            except OSError as exc:
                logger.warning("Could not delete %s: %s", path, exc)
        return freed

    @staticmethod
    def _safe_file_size(path: Path) -> Optional[int]:
        """Return file size in bytes, or None if the path is unreadable."""
        try:
            if path.is_file():
                return path.stat().st_size
        except OSError as exc:
            logger.warning("Skipping unreadable path %s: %s", path, exc)
        return None

    @staticmethod
    def _safe_iter_dir(directory: Path) -> List[Path]:
        """List directory entries, skipping unreadable paths."""
        try:
            return list(directory.iterdir())
        except OSError as exc:
            logger.warning("Could not list directory %s: %s", directory, exc)
            return []

    def get_disk_breakdown(self) -> Dict:
        """Filesystem breakdown for originals, scrubbed copies, and thumbnail cache."""
        originals_bytes = 0
        originals_count = 0
        for path in self._safe_iter_dir(self.uploads_dir):
            if path.name.startswith("."):
                continue
            size = self._safe_file_size(path)
            if size is None:
                continue
            originals_bytes += size
            originals_count += 1

        scrubbed_bytes = 0
        scrubbed_count = 0
        scrubbed_dir = self.uploads_dir / "scrubbed"
        if scrubbed_dir.is_dir():
            for path in self._safe_iter_dir(scrubbed_dir):
                if path.name == "privacy_audit.json":
                    continue
                size = self._safe_file_size(path)
                if size is None:
                    continue
                scrubbed_bytes += size
                scrubbed_count += 1

        thumbs_bytes = 0
        thumbs_count = 0
        thumbs_dir = self.uploads_dir / "thumbs"
        if thumbs_dir.is_dir():
            try:
                thumb_paths = thumbs_dir.rglob("*.jpg")
            except OSError as exc:
                logger.warning("Could not walk thumbnail cache %s: %s", thumbs_dir, exc)
                thumb_paths = []
            for path in thumb_paths:
                size = self._safe_file_size(path)
                if size is None:
                    continue
                thumbs_bytes += size
                thumbs_count += 1

        return {
            "originals": {
                "count": originals_count,
                "size_mb": originals_bytes / (1024 * 1024),
            },
            "scrubbed": {
                "count": scrubbed_count,
                "size_mb": scrubbed_bytes / (1024 * 1024),
            },
            "thumbnails": {
                "count": thumbs_count,
                "size_mb": thumbs_bytes / (1024 * 1024),
            },
        }

    # ──────────────────────────────────────────────────────────────────
    # Cleanup Operations
    # ──────────────────────────────────────────────────────────────────

    def cleanup_empty_images(self, dry_run: bool = False, days_grace: int = 30) -> Dict:
        """Delete empty image files (no animals) older than days_grace days."""
        conn = self.db_manager.get_connection()
        cursor = conn.cursor()
        cursor.execute(f'''
            SELECT id, filename, file_size_bytes
            FROM images
            WHERE file_tier = 'empty'
              AND file_status = 'available'
              AND datetime(uploaded_at) < datetime('now', '-{days_grace} days')
            LIMIT 10000
        ''')
        deletable = cursor.fetchall()
        conn.close()

        deleted_count = 0
        freed_bytes = 0
        errors = []

        for image_id, filename, file_size_bytes in deletable:
            try:
                if not dry_run:
                    self.delete_image_files(filename, image_id=image_id)
                    self.db_manager.delete_image_file(image_id)

                deleted_count += 1
                freed_bytes += file_size_bytes or 0
            except Exception as e:
                logger.warning(f"Could not delete image {image_id}: {e}")
                errors.append(str(e))

        return {
            "action": "delete_empty",
            "dry_run": dry_run,
            "deleted_count": deleted_count,
            "freed_mb": freed_bytes / (1024 * 1024),
            "errors": len(errors),
        }

    def cleanup_marked_for_deletion(self, days_grace: int = 7, dry_run: bool = False) -> Dict:
        """Delete files marked for deletion after grace period."""
        conn = self.db_manager.get_connection()
        cursor = conn.cursor()
        cursor.execute(f'''
            SELECT id, filename, file_size_bytes
            FROM images
            WHERE marked_for_deletion_at IS NOT NULL
            AND datetime(marked_for_deletion_at) < datetime('now', '-{days_grace} days')
            AND file_status = 'available'
            AND can_delete = 1
            LIMIT 10000
        ''')
        deletable = cursor.fetchall()
        conn.close()

        deleted_count = 0
        freed_bytes = 0
        errors = []

        for image_id, filename, file_size_bytes in deletable:
            try:
                if not dry_run:
                    self.delete_image_files(filename, image_id=image_id)
                    self.db_manager.delete_image_file(image_id)

                deleted_count += 1
                freed_bytes += file_size_bytes or 0
            except Exception as e:
                logger.warning(f"Could not delete image {image_id}: {e}")
                errors.append(str(e))

        return {
            "action": "cleanup_marked",
            "dry_run": dry_run,
            "deleted_count": deleted_count,
            "freed_mb": freed_bytes / (1024 * 1024),
            "grace_period_days": days_grace,
            "errors": len(errors),
        }

    def get_deletion_preview(self, tier: str, days_old: int = 7) -> Dict:
        """Preview what would be deleted."""
        conn = self.db_manager.get_connection()
        cursor = conn.cursor()

        if tier == "empty":
            cursor.execute('''
                SELECT COUNT(*), SUM(COALESCE(file_size_bytes, 0)), MIN(uploaded_at), MAX(uploaded_at)
                FROM images
                WHERE file_tier = 'empty' AND file_status = 'available'
            ''')
        else:
            cursor.execute('''
                SELECT COUNT(*), SUM(COALESCE(file_size_bytes, 0)), MIN(uploaded_at), MAX(uploaded_at)
                FROM images
                WHERE marked_for_deletion_at IS NOT NULL
                AND datetime(marked_for_deletion_at) < datetime('now', ?)
                AND file_status = 'available'
            ''', (f"-{days_old} days",))

        result = cursor.fetchone()
        conn.close()

        count = result[0] or 0
        total_size = result[1] or 0
        earliest = result[2]
        latest = result[3]

        return {
            "tier": tier,
            "will_delete": count,
            "total_size_mb": total_size / (1024 * 1024),
            "earliest_uploaded": earliest,
            "latest_uploaded": latest,
            "warning": f"These {count} files will be permanently deleted",
        }

    def cleanup_by_tier(
        self,
        tier: str,
        dry_run: bool = True,
        days_grace: int = 0,
    ) -> Dict:
        """Delete upload files for a tier (empty, low_conf, valid, all)."""
        conn = self.db_manager.get_connection()
        cursor = conn.cursor()
        age_clause = ""
        params: List = []
        if days_grace > 0:
            age_clause = "AND datetime(uploaded_at) < datetime('now', ?)"
            params.append(f"-{days_grace} days")

        tier_clause = ""
        if tier != "all":
            tier_clause = "AND file_tier = ?"
            params.insert(0, tier)

        cursor.execute(
            f"""
            SELECT id, filename, file_size_bytes
            FROM images
            WHERE file_status = 'available'
            {tier_clause}
            {age_clause}
            LIMIT 10000
            """,
            params,
        )
        deletable = cursor.fetchall()
        conn.close()

        deleted_count = 0
        freed_bytes = 0
        errors: List[str] = []

        for image_id, filename, file_size_bytes in deletable:
            try:
                if not dry_run:
                    freed = self.delete_image_files(filename, image_id=image_id)
                    self.db_manager.delete_image_file(image_id)
                    freed_bytes += freed
                else:
                    freed_bytes += file_size_bytes or 0
                deleted_count += 1
            except Exception as exc:
                logger.warning("Could not delete image %s: %s", image_id, exc)
                errors.append(str(exc))

        return {
            "action": "delete_by_tier",
            "tier": tier,
            "dry_run": dry_run,
            "deleted_count": deleted_count,
            "freed_mb": freed_bytes / (1024 * 1024),
            "days_grace": days_grace,
            "errors": len(errors),
        }

    def purge_thumbnails(self, dry_run: bool = True) -> Dict:
        """Delete the thumbnail cache tree (regenerates on next request)."""
        thumbs_dir = self.uploads_dir / "thumbs"
        freed_bytes = 0
        file_count = 0
        if thumbs_dir.is_dir():
            for path in thumbs_dir.rglob("*.jpg"):
                if path.is_file():
                    file_count += 1
                    freed_bytes += path.stat().st_size
            if not dry_run:
                shutil.rmtree(thumbs_dir, ignore_errors=True)
                thumbs_dir.mkdir(exist_ok=True)

        return {
            "action": "purge_thumbnails",
            "dry_run": dry_run,
            "deleted_count": file_count,
            "freed_mb": freed_bytes / (1024 * 1024),
            "errors": 0,
        }

    def purge_scrubbed_orphans(self, dry_run: bool = True) -> Dict:
        """Delete scrubbed files with no available DB record."""
        scrubbed_dir = self.uploads_dir / "scrubbed"
        if not scrubbed_dir.is_dir():
            return {
                "action": "purge_scrubbed_orphans",
                "dry_run": dry_run,
                "deleted_count": 0,
                "freed_mb": 0.0,
                "errors": 0,
            }

        conn = self.db_manager.get_connection()
        cursor = conn.cursor()
        cursor.execute(
            """
            SELECT filename FROM images
            WHERE file_status = 'available'
            """
        )
        available = {row[0] for row in cursor.fetchall()}
        conn.close()

        deleted_count = 0
        freed_bytes = 0
        errors = 0
        for path in scrubbed_dir.iterdir():
            if not path.is_file() or path.name == "privacy_audit.json":
                continue
            if path.name in available:
                continue
            try:
                freed_bytes += path.stat().st_size
                deleted_count += 1
                if not dry_run:
                    path.unlink()
            except OSError as exc:
                logger.warning("Could not delete scrubbed orphan %s: %s", path, exc)
                errors += 1

        return {
            "action": "purge_scrubbed_orphans",
            "dry_run": dry_run,
            "deleted_count": deleted_count,
            "freed_mb": freed_bytes / (1024 * 1024),
            "errors": errors,
        }

    def get_storage_status(self) -> Dict:
        """Get detailed storage breakdown."""
        stats = self.db_manager.get_storage_stats()

        breakdown = {}
        total_size = 0

        for tier, count, total_bytes, oldest in stats or []:
            size_mb = (total_bytes or 0) / (1024 * 1024)
            breakdown[tier] = {
                "count": count,
                "size_mb": size_mb,
                "oldest_upload": oldest,
            }
            total_size += size_mb

        return {
            "total_mb": total_size,
            "breakdown": breakdown,
            "disk": self.get_disk_breakdown(),
            "timestamp": datetime.utcnow().isoformat(),
        }

    def get_deletion_warnings(self) -> Dict:
        """Get images pending deletion (user warnings)."""
        images = self.db_manager.get_images_for_deletion_warning(days_until_deletion=3)

        return {
            "pending_deletion_count": len(images),
            "images": [
                {
                    "id": img[0],
                    "filename": img[1],
                    "marked_for_deletion_at": img[2],
                }
                for img in images
            ],
        }

    # ──────────────────────────────────────────────────────────────────
    # Hash Management (for performance optimization)
    # ──────────────────────────────────────────────────────────────────

    def get_hash_optimization_stats(self) -> Dict:
        """Get hash statistics and optimization opportunities."""
        stats = self.db_manager.get_hash_stats()
        duplicates = self.db_manager.find_duplicate_files()

        return {
            "hashes": stats,
            "duplicates": [
                {
                    "hash": dup[0][:16] + "..." if dup[0] else None,
                    "count": dup[1],
                    "image_ids": [int(x) for x in dup[2].split(",")],
                }
                for dup in (duplicates or [])
            ],
            "potential_savings_mb": sum(
                (dup[1] - 1) * 1.5 for dup in (duplicates or [])
            ),  # ~1.5 MB per image estimate
        }

    def clear_hashes_for_optimization(self, strategy: str) -> Dict:
        """
        Clear hashes to reduce processing time.

        Strategies:
        - 'empty': Clear hashes for empty tier (deleted files)
        - 'archived': Clear hashes for archived tier
        - 'old_30d': Clear hashes older than 30 days
        - 'duplicates': Keep only first occurrence of duplicate hashes
        - 'all': Clear all hashes
        """
        if strategy == "empty":
            count = self.db_manager.clear_hashes_by_tier("empty")
            return {"strategy": strategy, "cleared": count, "purpose": "Removed hashes from empty frames"}

        elif strategy == "archived":
            count = self.db_manager.clear_hashes_by_status("archived")
            return {"strategy": strategy, "cleared": count, "purpose": "Removed hashes from archived images"}

        elif strategy == "old_30d":
            count = self.db_manager.clear_old_hashes(days_old=30)
            return {"strategy": strategy, "cleared": count, "purpose": "Removed hashes older than 30 days"}

        elif strategy == "duplicates":
            count = self.db_manager.clear_duplicate_hashes()
            return {
                "strategy": strategy,
                "cleared": count,
                "purpose": "Kept only first occurrence of duplicate hashes",
            }

        elif strategy == "all":
            count = self.db_manager.clear_all_hashes()
            return {
                "strategy": strategy,
                "cleared": count,
                "purpose": "Removed all hashes (maximum performance gain)",
                "warning": "Deduplication detection disabled until new hashes calculated",
            }

        else:
            raise ValueError(f"Unknown strategy: {strategy}")


# Singleton instance (will be initialized in backend.main)
file_manager: Optional[FileManager] = None
