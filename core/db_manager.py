
import sqlite3
import pandas as pd
import json
from datetime import datetime
from pathlib import Path

class DatabaseManager:
    def __init__(self, db_path="wildlife_data.db"):
        self.db_path = db_path
        self._init_db()

    def get_connection(self):
        return sqlite3.connect(self.db_path)

    def _init_db(self):
        """Initialize database with required tables."""
        conn = self.get_connection()
        cursor = conn.cursor()
        
        # Images table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS images (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                filename TEXT NOT NULL,
                processed_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                capture_date TEXT,
                capture_time TEXT,
                temperature TEXT,
                day_night TEXT,
                brightness REAL,
                user_notes TEXT
            )
        ''')
        
        # Detections table (linked to images)
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS detections (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                image_id INTEGER,
                detected_animal TEXT,
                confidence REAL,
                method TEXT,
                bbox TEXT,
                FOREIGN KEY (image_id) REFERENCES images (id)
            )
        ''')
        
        conn.commit()
        conn.close()

    def save_results(self, df):
        """
        Save processing results to database.
        
        Args:
            df (pd.DataFrame): DataFrame containing processing results
            
        Returns:
            int: Number of images saved
        """
        if df is None or len(df) == 0:
            return 0
            
        conn = self.get_connection()
        cursor = conn.cursor()
        
        count = 0
        try:
            for _, row in df.iterrows():
                # Insert into images
                cursor.execute('''
                    INSERT INTO images (
                        filename, capture_date, capture_time, temperature, 
                        day_night, brightness, user_notes
                    ) VALUES (?, ?, ?, ?, ?, ?, ?)
                ''', (
                    row['filename'],
                    row.get('date'),
                    row.get('time'),
                    row.get('temperature'),
                    row.get('day_night'),
                    row.get('brightness', 0.0),
                    row.get('user_notes', '')
                ))
                
                image_id = cursor.lastrowid
                
                # Insert into detections
                # Handle bbox serialization
                bbox_json = json.dumps(row.get('bbox')) if row.get('bbox') else None
                
                cursor.execute('''
                    INSERT INTO detections (
                        image_id, detected_animal, confidence, method, bbox
                    ) VALUES (?, ?, ?, ?, ?)
                ''', (
                    image_id,
                    row['detected_animal'],
                    row.get('detection_confidence', 0.0),
                    row.get('detection_method', 'Unknown'),
                    bbox_json
                ))
                
                count += 1
            
            conn.commit()
            return count
            
        except Exception as e:
            conn.rollback()
            print(f"Error saving to database: {e}")
            raise e
        finally:
            conn.close()

    def get_history_df(self):
        """Retrieve full history as a flat DataFrame."""
        conn = self.get_connection()
        
        query = '''
            SELECT 
                i.id, i.filename, i.processed_at, i.capture_date, i.capture_time, 
                i.temperature, i.day_night, i.brightness, i.user_notes,
                d.detected_animal, d.confidence as detection_confidence, 
                d.method as detection_method, d.bbox
            FROM images i
            JOIN detections d ON i.id = d.image_id
            ORDER BY i.processed_at DESC
        '''
        
        try:
            df = pd.read_sql_query(query, conn)
            # Parse bbox back to list/None if needed, but for display string is fine.
            return df
        except Exception as e:
            print(f"Error fetching history: {e}")
            return pd.DataFrame()
        finally:
            conn.close()

    def clear_history(self):
        """Clear all data from database."""
        conn = self.get_connection()
        cursor = conn.cursor()
        cursor.execute("DELETE FROM detections")
        cursor.execute("DELETE FROM images")
        conn.commit()
        conn.close()

