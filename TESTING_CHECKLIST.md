# File Management System - Testing Checklist

## Pre-Deployment Verification

### ✅ Database & Backend
- [ ] `python3 -m py_compile backend/services/file_manager.py`
- [ ] `python3 -m py_compile backend/routers/storage.py`
- [ ] `python3 -m py_compile backend/routers/images.py`
- [ ] `python3 -m py_compile backend/routers/results.py`
- [ ] `python3 -m py_compile core/db_manager.py`

### ✅ Frontend Build
- [ ] `cd frontend && npm run build` (no TypeScript errors)
- [ ] Check browser console for errors when running dev server

---

## Integration Testing (Manual)

### 1️⃣ File Upload & Tier Classification

**Setup**: Fresh upload of 10 test images
- 3 with animals (confidence > 0.4) → **valid** tier
- 4 with low confidence (0.2-0.4) → **low_conf** tier
- 2 with Person/Vehicle → **empty** tier
- 1 completely empty → **empty** tier

**Tests**:
- [ ] Upload images via UI
- [ ] Check images appear in results
- [ ] Verify file tiers are correct via DB query:
  ```sql
  SELECT filename, file_tier, file_status, has_animal FROM images ORDER BY id DESC LIMIT 10;
  ```

### 2️⃣ Hash Tracking

**Tests**:
- [ ] Navigate to Storage page
- [ ] Verify hash stats appear
- [ ] Check: "Total images", "With hashes", "Unique hashes"
- [ ] Verify no duplicate detection on first upload
- [ ] Upload same image again
- [ ] Check duplicates are detected in stats

### 3️⃣ Hash Clearing Strategies

**Tests**:
- [ ] Click "Clear Empty Hashes" → verify count
- [ ] Click "Deduplicate Hashes" → verify duplicate hashes removed
- [ ] Click "Clear Old (30d+)" → verify old hashes cleared
- [ ] Click "Clear All Hashes" → verify all removed

### 4️⃣ Storage Status & Breakdown

**Tests**:
- [ ] Storage gauge shows correct total MB
- [ ] Tier breakdown shows correct counts per tier
- [ ] Storage percentage calculation is accurate
- [ ] Warning appears at 70% threshold (if applicable)
- [ ] Critical warning at 90% (if applicable)

### 5️⃣ Batch Downloads

**Tests**:
- [ ] Click "Download All" for empty tier
- [ ] ZIP file downloads successfully
- [ ] ZIP contains:
  - [ ] `images/` folder with actual files
  - [ ] `metadata.json` with image info
  - [ ] `README.txt` with manifest
- [ ] Repeat for low_conf and valid tiers
- [ ] Test batch download API:
  ```bash
  curl -X POST "http://localhost:8000/api/storage/downloads/batch?tier=empty"
  ```

### 6️⃣ Export Tracking

**Tests**:
- [ ] Export to Excel
- [ ] Check detections marked as exported:
  ```sql
  SELECT COUNT(*) as exported FROM detections WHERE is_exported = 1;
  ```
- [ ] Export to CSV → verify same detections marked exported
- [ ] Export to Darwin Core → verify same detections marked
- [ ] Export to Wildlife Insights → verify same detections marked

### 7️⃣ Result Deletion

**Tests**:
- [ ] Go to Results page
- [ ] Delete a single result:
  ```bash
  curl -X DELETE "http://localhost:8000/api/results/1"
  ```
- [ ] Verify result disappears from UI
- [ ] Check image is marked for deletion:
  ```sql
  SELECT id, marked_for_deletion_at FROM images WHERE id = [image_id];
  ```
- [ ] Batch delete multiple results:
  ```bash
  curl -X DELETE "http://localhost:8000/api/results?detection_ids=2,3,4"
  ```

### 8️⃣ File Cleanup

**Tests**:
- [ ] Manual cleanup: `GET /api/storage/cleanup?action=delete_empty&dry_run=true`
- [ ] Verify deletion preview is correct
- [ ] Execute cleanup: `GET /api/storage/cleanup?action=delete_empty&dry_run=false`
- [ ] Check files are deleted from `/uploads` directory
- [ ] Verify DB records show `file_status = 'deleted'`
- [ ] Metadata still queryable:
  ```sql
  SELECT id, filename FROM images WHERE file_status = 'deleted' LIMIT 5;
  ```

### 9️⃣ Background Cleanup Scheduler

**Tests**:
- [ ] Check backend logs for "Running scheduled file cleanup"
- [ ] Verify cleanup runs hourly
- [ ] Mark an old image for deletion:
  ```sql
  UPDATE images SET marked_for_deletion_at = datetime('now', '-8 days') WHERE id = [test_id];
  ```
- [ ] Wait for cleanup to run or trigger manually
- [ ] Verify file is deleted

### 🔟 Person/Vehicle Tier Classification

**Tests**:
- [ ] Upload image with Person detection
- [ ] Verify `file_tier = 'empty'` in database
- [ ] Upload image with Vehicle detection
- [ ] Verify `file_tier = 'empty'` in database
- [ ] Upload image with both Person and Animal detections
- [ ] Verify treated as empty (Person prioritized)

### 1️⃣1️⃣ Storage Dashboard UI

**Tests**:
- [ ] Check loading skeleton appears briefly
- [ ] Error state shows when API fails (simulate with network tab)
- [ ] Retry button works
- [ ] All sections load correctly
- [ ] Numbers update when data refreshes
- [ ] Dark mode styling works

### 1️⃣2️⃣ Edge Cases

**Tests**:
- [ ] Upload image > 50 MB → verify rejected
- [ ] Upload non-image file → verify rejected
- [ ] Download empty tier → verify creates valid ZIP
- [ ] Delete all detections for an image → mark image for deletion
- [ ] Clear hashes with no hashes to clear → handle gracefully
- [ ] Access storage page without auth (if applicable) → proper error

---

## Performance Testing

### Query Performance

```sql
-- Should complete < 100ms
SELECT file_tier, COUNT(*), SUM(file_size_bytes) FROM images 
GROUP BY file_tier;

-- Should complete < 500ms
SELECT id, filename FROM images 
WHERE file_tier = 'empty' AND file_status = 'available' 
LIMIT 10000;

-- Should complete < 1s
SELECT file_hash, COUNT(*) FROM images 
WHERE file_hash IS NOT NULL 
GROUP BY file_hash 
HAVING COUNT(*) > 1;
```

### ZIP Creation Performance

- [ ] 100 images: < 5 seconds
- [ ] 1000 images: < 30 seconds
- [ ] Verify streaming doesn't buffer entire ZIP in memory

### Memory Usage

- [ ] Monitor backend memory during cleanup operation
- [ ] Monitor frontend memory during large table render
- [ ] Check for memory leaks with extended usage

---

## Deployment Checklist

### Before Going Live

- [ ] All tests pass
- [ ] No Python syntax errors
- [ ] No TypeScript errors
- [ ] Database migrations applied
- [ ] Backup existing database
- [ ] Test rollback plan
- [ ] Environment variables set correctly:
  - [ ] `DB_PATH`
  - [ ] `MAX_UPLOAD_MB`
  - [ ] `JOB_TTL_HOURS`
  - [ ] `CORS_ORIGINS`

### Deployment Steps

```bash
# 1. Stop existing services
systemctl stop camera-traps

# 2. Backup database
cp wildlife_data.db wildlife_data.db.backup

# 3. Deploy code
git pull origin main
cd frontend && npm install && npm run build
cd ..

# 4. Start services
systemctl start camera-traps

# 5. Verify services are running
curl http://localhost:8000/api/config

# 6. Run smoke tests
# - Upload 1 image
# - Export results
# - Check storage page loads
# - Verify cleanup runs
```

### Post-Deployment

- [ ] Monitor backend logs for errors
- [ ] Test critical workflows:
  - [ ] Upload image
  - [ ] Process with AI
  - [ ] Review results
  - [ ] Export results
  - [ ] Download files
- [ ] Check storage page loads
- [ ] Verify cleanup scheduler is running
- [ ] Monitor disk usage

---

## Rollback Plan

If critical issues discovered:

```bash
# 1. Restore backup
cp wildlife_data.db.backup wildlife_data.db

# 2. Revert code
git revert [commit-hash]
cd frontend && npm run build

# 3. Restart services
systemctl restart camera-traps

# 4. Verify rollback successful
curl http://localhost:8000/api/config
```

---

## Known Limitations & Future Improvements

### Current Implementation

✅ Supports up to 10,000 files per cleanup operation  
✅ ZIP streaming for large downloads  
✅ Efficient tier-based cleanup  
✅ Hash deduplication tracking  
✅ Soft deletes with audit trail  

### Future Enhancements

- [ ] Compression for archived tier (optional)
- [ ] S3/cloud storage integration
- [ ] Incremental hash calculation
- [ ] Advanced filtering/search in dashboard
- [ ] Quota enforcement per station/project
- [ ] Email alerts for storage warnings
- [ ] API rate limiting

---

## Support & Troubleshooting

### Common Issues

**ZIP download hangs:**
- Check backend logs for errors
- Verify disk space available
- Check file permissions on uploads directory

**Cleanup doesn't run:**
- Check backend logs for scheduler
- Verify database has images to delete
- Check `marked_for_deletion_at` timestamps

**High memory usage:**
- Reduce `_PARALLEL_IMAGES` in images.py
- Reduce ZIP chunk size in file_manager.py
- Monitor with `free -h` and `ps aux`

**Slow exports:**
- Check database indexes on detections table
- Consider archiving old records to separate DB
- Profile with `EXPLAIN QUERY PLAN`

---

## Sign-Off

- [ ] QA Tested: _________________ Date: _______
- [ ] Deployment Ready: _________ Date: _______
- [ ] Post-Deployment Verified: __ Date: _______
