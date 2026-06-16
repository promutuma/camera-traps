# File Management System - Quick Reference

## For Developers

### Core Workflows

**Uploading Images**
```python
# Images auto-classified into tiers:
# - 'empty' (no animals, Person, Vehicle, confidence=0)
# - 'low_conf' (0.2-0.4 confidence)
# - 'valid' (>0.4 confidence)
```

**Checking Storage Usage**
```bash
curl http://localhost:8000/api/storage/status
# Returns: {breakdown: {empty: {...}, low_conf: {...}, valid: {...}}}
```

**Downloading Files Before Deletion**
```bash
# Create batch download
curl -X POST "http://localhost:8000/api/storage/downloads/batch?tier=empty"
# Returns: {download_id: "abc123", status: "preparing", ...}

# Download ZIP
curl "http://localhost:8000/api/storage/downloads/abc123/stream" > files.zip
```

**Exporting Results** (marks images as exported)
```bash
# All exports automatically set is_exported = TRUE
# Enables aggressive cleanup of images after export
GET /api/results/export/excel
GET /api/results/export/csv
GET /api/exports/darwin-core
GET /api/exports/wildlife-insights
```

**Cleaning Up Files**
```bash
# Preview what would be deleted
curl "http://localhost:8000/api/storage/deletion-preview?tier=empty&days_old=7"

# Execute cleanup (production)
curl -X POST "http://localhost:8000/api/storage/cleanup?action=delete_empty&dry_run=false"

# Preview only (safe)
curl -X POST "http://localhost:8000/api/storage/cleanup?action=delete_empty&dry_run=true"
```

**Managing Hashes**
```bash
# View hash statistics
curl http://localhost:8000/api/storage/hash-stats

# Clear empty frame hashes
curl -X POST "http://localhost:8000/api/storage/clear-hashes?strategy=empty"

# Clear all hashes (maximum performance)
curl -X POST "http://localhost:8000/api/storage/clear-hashes?strategy=all"
```

**Deleting Results**
```bash
# Single deletion
DELETE /api/results/123

# Batch deletion
DELETE /api/results?detection_ids=123,124,125
```

---

## For DevOps/Operations

### Monitoring

**Storage Metrics**
```bash
# Check DB size
du -sh wildlife_data.db

# Check uploads directory
du -sh uploads/

# Query storage by tier
sqlite3 wildlife_data.db << EOF
SELECT file_tier, COUNT(*), SUM(file_size_bytes)/1024/1024 as MB 
FROM images WHERE file_status='available' 
GROUP BY file_tier;
EOF
```

**Cleanup Status**
```bash
# Check backend logs for cleanup runs
tail -f /var/log/camera-traps/app.log | grep "cleanup"

# Verify scheduled cleanup is configured
# (runs every hour automatically)
```

**File Count by Tier**
```bash
sqlite3 wildlife_data.db << EOF
SELECT 'empty' as tier, COUNT(*) FROM images WHERE file_tier='empty'
UNION ALL
SELECT 'low_conf', COUNT(*) FROM images WHERE file_tier='low_conf'
UNION ALL
SELECT 'valid', COUNT(*) FROM images WHERE file_tier='valid'
UNION ALL
SELECT 'deleted', COUNT(*) FROM images WHERE file_status='deleted';
EOF
```

### Maintenance Tasks

**Weekly**
- [ ] Monitor `du -sh uploads/` - should not grow unbounded
- [ ] Check backend logs for errors
- [ ] Verify cleanup runs completed

**Monthly**
- [ ] Backup database: `cp wildlife_data.db wildlife_data.db.backup`
- [ ] Review disk usage trends
- [ ] Check for orphaned temp directories

**Quarterly**
- [ ] Analyze deletion patterns
- [ ] Review hash statistics
- [ ] Optimize queries if needed

### Emergency Procedures

**Disk Space Crisis** (>95% full)
```bash
# 1. Check what's consuming space
du -sh uploads/
du -sh wildlife_data.db

# 2. Force cleanup immediately
curl -X POST "http://localhost:8000/api/storage/cleanup?action=delete_empty&dry_run=false"
curl -X POST "http://localhost:8000/api/storage/cleanup?action=delete_marked&days_old=0&dry_run=false"

# 3. Clear all non-critical hashes
curl -X POST "http://localhost:8000/api/storage/clear-hashes?strategy=all"

# 4. Verify space freed
df -h
```

**Restore from Backup**
```bash
# Stop application
systemctl stop camera-traps

# Restore database
cp wildlife_data.db.backup wildlife_data.db

# Restore files (if backed up)
# cp -r uploads.backup/* uploads/

# Restart
systemctl start camera-traps

# Verify
curl http://localhost:8000/api/config
```

---

## For End Users

### Storage Management UI

**Dashboard Shows:**
- 📊 Total storage usage (GB)
- 📈 Storage percentage (with warnings at 70%/90%)
- 🎯 Breakdown by tier (empty, low-conf, valid)
- ⏰ Deletion countdown for pending files
- 📥 Download buttons for each tier
- 🔄 Hash deduplication stats

**To Download Files Before Deletion:**
1. Go to Storage Management page
2. Click "Download All" for any tier
3. ZIP file downloads automatically
4. Contains images + metadata.json

**To Optimize Storage:**
1. Go to Hash Management section
2. Choose clearing strategy:
   - **Clear Empty Hashes**: Removes from unused frames
   - **Deduplicate Hashes**: Removes duplicate tracking
   - **Clear Old (30d+)**: Removes from old images
   - **Clear All Hashes**: Maximum speed gain
3. Confirm and execute

---

## Configuration

### Environment Variables
```bash
DB_PATH=wildlife_data.db          # Database location
MAX_UPLOAD_MB=50                  # Max upload size
JOB_TTL_HOURS=2                   # Job retention time
CORS_ORIGINS=...                  # CORS whitelist
```

### Key Settings in Code
```python
# backend/routers/images.py
_PARALLEL_IMAGES = 2              # Images processed in parallel
_MAX_UPLOAD_BYTES = 50 * 1024**2  # 50 MB default

# backend/services/job_manager.py
_TTL_SECONDS = 2 * 3600           # 2 hour job retention

# backend/main.py
# Cleanup runs every 3600 seconds (1 hour)
```

---

## FAQ

**Q: How much space can I free by deleting empty images?**  
A: Typically 70-80% of total storage. Empty images have no value and can be deleted aggressively.

**Q: What happens to results after I delete an image?**  
A: Results (Excel, CSV, DNA Core) are preserved. Only the image file is deleted, not the data.

**Q: Can I recover deleted images?**  
A: Images deleted via cleanup cannot be recovered. Downloads are available for 7 days before deletion. Always download before deadline!

**Q: How does hash clearing affect deduplication?**  
A: Clearing hashes disables duplicate detection until new uploads recalculate them. New images will be hashed normally.

**Q: When does cleanup run?**  
A: Automatically every hour in the background. Can also be triggered manually via API.

**Q: Is it safe to use "Delete All Hashes"?**  
A: Yes, but you'll lose duplicate detection until new files are uploaded and hashed.

**Q: What's the difference between "marked_for_deletion" and "deleted"?**  
A: Marked = scheduled to delete after 7-day grace period. Deleted = file gone, metadata kept.

**Q: Can legal holds prevent deletion?**  
A: Yes, set `can_delete = FALSE` on images to prevent auto-deletion permanently.

---

## Troubleshooting

**Downloads hanging**
- Check backend logs for errors
- Verify disk space available
- Reduce batch size if downloading >1000 images

**Cleanup not running**
- Check backend logs: `grep cleanup app.log`
- Verify database has images marked for deletion
- Restart backend service

**Storage page won't load**
- Check browser console for errors
- Verify `/api/storage/status` endpoint responds
- Clear browser cache and refresh

**ZIP file corrupted**
- Verify all files downloaded completely
- Check network stability during download
- Try downloading smaller batch size

---

## Performance Tips

1. **Clear hashes regularly** - Reduces processing overhead
2. **Download files proactively** - Don't wait until last day
3. **Schedule exports off-peak** - Avoid during cleanup
4. **Monitor disk usage weekly** - Prevent emergency situations
5. **Archive results** - Keep exports, delete images after 30 days

---

## Support Matrix

| Issue | Check | Solution |
|-------|-------|----------|
| High disk usage | Storage page | Delete empty tier images |
| Slow uploads | Backend logs | Reduce parallel images setting |
| Missing files | Deletion preview | Download during grace period |
| API errors | Backend logs | Verify database connectivity |
| ZIP download fails | Browser console | Check network, retry download |

---

**Version**: 1.0  
**Last Updated**: 2026-06-16  
**Status**: Production Ready ✅
