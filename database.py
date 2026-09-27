import sqlite3
import json
import os
from datetime import datetime

class DatabaseManager:
    def __init__(self, db_name="security_v2.db"):
        self.db_name = db_name
        self.create_tables()

    def get_connection(self):
        return sqlite3.connect(self.db_name)

    def create_tables(self):
        conn = self.get_connection()
        cursor = conn.cursor()

        # Users table (Updated)
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT NOT NULL,
                age INTEGER,
                gender TEXT,
                department TEXT,
                batch_no TEXT,
                authority_rank INTEGER DEFAULT 1,
                face_image_path TEXT,
                embedding TEXT -- Stored as JSON string
            )
        ''')

        # Rooms table (Updated)
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS rooms (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                room_name TEXT NOT NULL UNIQUE,
                allowed_ranks TEXT DEFAULT "1" -- Comma separated "1,2,3"
            )
        ''')

        # Access Logs table (Updated)
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS access_logs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                person_id INTEGER,
                person_name TEXT,
                room_name TEXT,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                status TEXT DEFAULT 'AUTHORIZED', -- AUTHORIZED / UNAUTHORIZED
                confidence REAL
            )
        ''')

        # Alerts table (Optional but good)
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS alerts (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                message TEXT,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                read_status INTEGER DEFAULT 0
            )
        ''')
        
        # Initialize default rooms if empty
        cursor.execute('SELECT COUNT(*) FROM rooms')
        if cursor.fetchone()[0] == 0:
            cursor.execute('INSERT INTO rooms (room_name, allowed_ranks) VALUES (?, ?)', ("Room 1", "1,2"))
            cursor.execute('INSERT INTO rooms (room_name, allowed_ranks) VALUES (?, ?)', ("Room 2", "1,2,3"))
            # New Room Names requested by user
            cursor.execute('INSERT INTO rooms (room_name, allowed_ranks) VALUES (?, ?)', ("block1", "1,2"))
            cursor.execute('INSERT INTO rooms (room_name, allowed_ranks) VALUES (?, ?)', ("block 2", "1,2,3"))
            cursor.execute('INSERT INTO rooms (room_name, allowed_ranks) VALUES (?, ?)', ("jection1", "1,2,3"))
            
        conn.commit()
        conn.close()

    # --- User Management ---
    def add_user(self, name, age, gender, department, batch_no, authority_rank, face_image_path, embedding):
        conn = self.get_connection()
        cursor = conn.cursor()
        cursor.execute('''
            INSERT INTO users (name, age, gender, department, batch_no, authority_rank, face_image_path, embedding)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        ''', (name, age, gender, department, batch_no, authority_rank, face_image_path, json.dumps(embedding)))
        conn.commit()
        conn.close()
        return cursor.lastrowid

    def get_users(self):
        conn = self.get_connection()
        cursor = conn.cursor()
        cursor.execute('SELECT * FROM users')
        rows = cursor.fetchall()
        users = []
        columns = [column[0] for column in cursor.description]
        for row in rows:
            users.append(dict(zip(columns, row)))
        conn.close()
        return users

    def delete_user(self, user_id):
        conn = self.get_connection()
        cursor = conn.cursor()
        cursor.execute('DELETE FROM users WHERE id = ?', (user_id,))
        conn.commit()
        conn.close()

    def get_user_embeddings(self):
        """Returns {name: {'embedding': list, 'rank': int, 'id': int}}"""
        conn = self.get_connection()
        cursor = conn.cursor()
        cursor.execute('SELECT id, name, authority_rank, embedding FROM users')
        rows = cursor.fetchall()
        data = {}
        for uid, name, rank, emb_json in rows:
            if emb_json:
                data[name] = {
                    'id': uid,
                    'embedding': json.loads(emb_json),
                    'rank': rank
                }
        conn.close()
        return data

    # --- Room Management ---
    def get_rooms(self):
        conn = self.get_connection()
        cursor = conn.cursor()
        cursor.execute('SELECT * FROM rooms')
        rows = cursor.fetchall()
        rooms = []
        columns = [column[0] for column in cursor.description]
        for row in rows:
            rooms.append(dict(zip(columns, row)))
        conn.close()
        return rooms

    def update_room_ranks(self, room_id, allowed_ranks_str):
        conn = self.get_connection()
        cursor = conn.cursor()
        cursor.execute('UPDATE rooms SET allowed_ranks = ? WHERE id = ?', (allowed_ranks_str, room_id))
        conn.commit()
        conn.close()

    def clear_logs(self):
        conn = self.get_connection()
        cursor = conn.cursor()
        cursor.execute('DELETE FROM access_logs')
        cursor.execute('DELETE FROM alerts')
        conn.commit()
        conn.close()

    # --- Logging ---
    def log_access(self, person_id, person_name, room_name, status, confidence=0.0):
        # Prevent spamming logs: Check if last log for same person/room/status was < 5 sec ago
        conn = self.get_connection()
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT timestamp FROM access_logs 
            WHERE person_name = ? AND room_name = ? AND status = ? 
            ORDER BY id DESC LIMIT 1
        ''', (person_name, room_name, status))
        row = cursor.fetchone()
        
        should_log = True
        if row:
            last_time = datetime.strptime(row[0], "%Y-%m-%d %H:%M:%S")
            if (datetime.now() - last_time).total_seconds() < 5:
                should_log = False
        
        log_id = None
        if should_log:
            cursor.execute('''
                INSERT INTO access_logs (person_id, person_name, room_name, status, confidence)
                VALUES (?, ?, ?, ?, ?)
            ''', (person_id, person_name, room_name, status, confidence))
            log_id = cursor.lastrowid
            conn.commit()
            
            # If unauthorized, add alert
            if status == "UNAUTHORIZED":
                msg = f"Unauthorized access attempt by {person_name} in {room_name}"
                cursor.execute('INSERT INTO alerts (message) VALUES (?)', (msg,))
                conn.commit()

        conn.close()
        return log_id

    def get_logs(self, limit=50):
        conn = self.get_connection()
        cursor = conn.cursor()
        cursor.execute('SELECT * FROM access_logs ORDER BY timestamp DESC LIMIT ?', (limit,))
        rows = cursor.fetchall()
        logs = []
        columns = [column[0] for column in cursor.description]
        for row in rows:
            logs.append(dict(zip(columns, row)))
        conn.close()
        return logs

    def get_alerts(self, limit=10):
        conn = self.get_connection()
        cursor = conn.cursor()
        cursor.execute('SELECT * FROM alerts ORDER BY timestamp DESC LIMIT ?', (limit,))
        rows = cursor.fetchall()
        alerts = []
        columns = [column[0] for column in cursor.description]
        for row in rows:
            alerts.append(dict(zip(columns, row)))
        conn.close()
        return alerts

    def update_log_person(self, log_id, person_id, person_name, status):
        """Updates the person and status of an existing log entry."""
        if log_id is None: return
        conn = self.get_connection()
        cursor = conn.cursor()
        cursor.execute('''
            UPDATE access_logs 
            SET person_id = ?, person_name = ?, status = ?
            WHERE id = ?
        ''', (person_id, person_name, status, log_id))
        conn.commit()
        conn.close()
