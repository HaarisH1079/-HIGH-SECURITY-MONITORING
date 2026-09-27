import tkinter as tk #
from tkinter import ttk, messagebox, simpledialog
import cv2
from PIL import Image, ImageTk
import datetime
import time
import threading
import os
import sys
import json
import numpy as np
from database import DatabaseManager
from recognition import FaceRecognitionEngine
from room_detector import RoomDetector
import queue
import concurrent.futures

# --- Multi-Camera Manager ---
# All cameras MUST open in the same process because Windows MSMF assigns
# camera indices dynamically per-session. A subprocess sees different indices.
# The fix for Camera 2 (which shares a USB host): open all cameras first,
# then warm-up Camera 2 with grab() calls to flush the MSMF hardware buffer.

class CameraSlot:
    """A thread-safe slot holding the latest frame for one camera."""
    def __init__(self, index, name):
        self.index = index
        self.name = name
        self.cap = None
        self.frame = None
        self.ret = False
        self._lock = threading.Lock()
        self._running = False
        self._thread = None

    def _grab_loop(self):
        fail = 0
        while self._running:
            if self.cap is None or not self.cap.isOpened():
                time.sleep(0.5)
                continue
            ret, frame = self.cap.read()
            if ret and frame is not None:
                with self._lock:
                    self.ret = True
                    self.frame = cv2.flip(frame, 1)
                fail = 0
            else:
                fail += 1
                if fail % 60 == 0:
                    print(f"[Cam {self.index}] {self.name}: frame drop #{fail}")
                if fail > 120:
                    print(f"[Cam {self.index}] {self.name}: too many drops, releasing cap")
                    self.cap.release()
                    self.cap = None
                    fail = 0

    def start_thread(self):
        self._running = True
        self._thread = threading.Thread(
            target=self._grab_loop, daemon=True, name=f"Cam-{self.index}-{self.name}"
        )
        self._thread.start()

    def read(self):
        with self._lock:
            return self.ret, self.frame.copy() if self.frame is not None else None

    def isOpened(self):
        return self.cap is not None and self.cap.isOpened()

    def release(self):
        self._running = False
        if self.cap:
            self.cap.release()
            self.cap = None


class MultiCameraManager:
    """
    Opens all cameras in one MSMF session with staggered timing.
    Camera 2 (USB-hub conflict) gets a warm-up flush of grab() calls
    to clear the MSMF hardware-start error before .read() is called.
    """
    CAM_CONFIGS = [
        (0, "Laptop Cam"),
        (1, "External Cam 1"),
    ]
    OPEN_DELAY     = 2.5   # Increased delay for hardware stability
    TARGET_WIDTH   = 640
    TARGET_HEIGHT  = 480

    def __init__(self):
        self.slots = []
        self._open_all()

    def _open_all(self):
        caps = []
        for idx, name in self.CAM_CONFIGS:
            # Prioritize MSMF (1400) as Laptop Cam works well with it
            # and it often handles resolution/MJPG better on modern Windows
            backend_order = [cv2.CAP_MSMF, cv2.CAP_DSHOW, cv2.CAP_ANY]
            
            cap = None
            used_backend = -1
            
            for backend in backend_order:
                print(f"[MultiCamMgr] Trying {name} (idx={idx}) with backend {backend}...")
                cap = cv2.VideoCapture(idx, backend)
                if cap.isOpened():
                    cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.TARGET_WIDTH)
                    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.TARGET_HEIGHT)
                    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
                    
                    # Read-test: Ensure frames are actually coming
                    res, test_frame = cap.read()
                    if res and test_frame is not None:
                        used_backend = backend
                        print(f"[MultiCamMgr] {name} (idx={idx}) Success with backend {backend}")
                        break
                    else:
                        print(f"[MultiCamMgr] {name} (idx={idx}) backend {backend} opened but NO FRAMES.")
                        cap.release()
                        cap = None

            caps.append((idx, name, cap if (cap and cap.isOpened()) else None))
            time.sleep(self.OPEN_DELAY)

        # Create slots and start threads
        for idx, name, cap in caps:
            slot = CameraSlot(idx, name)
            slot.cap = cap
            slot.start_thread()
            self.slots.append(slot)

    def get_slots(self):
        return self.slots

    def release_all(self):
        for slot in self.slots:
            slot.release()


class RecognitionWorker:
    """Handles face recognition in a background thread to prevent UI freezing."""
    def __init__(self, recognizer):
        self.recognizer = recognizer
        self.task_queue = queue.Queue(maxsize=5)
        self.results_queue = queue.Queue()
        self._running = True
        self._thread = threading.Thread(target=self._process_loop, daemon=True)
        self._thread.start()

    def _process_loop(self):
        while self._running:
            try:
                # task: (cam_idx, objectID, face_img, context)
                cam_idx, objectID, face_img, context = self.task_queue.get(timeout=0.5)
                name, conf = self.recognizer.recognize_face(face_img)
                self.results_queue.put((cam_idx, objectID, name, conf, context))
                self.task_queue.task_done()
            except queue.Empty:
                continue
            except Exception as e:
                print(f"[RecognitionWorker] Error: {e}")

    def submit(self, cam_idx, objectID, face_img, context=None):
        if not self.task_queue.full():
            try:
                self.task_queue.put_nowait((cam_idx, objectID, face_img, context))
                return True
            except queue.Full:
                pass
        return False

    def get_results(self):
        results = []
        while not self.results_queue.empty():
            try:
                results.append(self.results_queue.get_nowait())
            except queue.Empty:
                break
        return results

    def stop(self):
        self._running = False

# --- Constants & Config ---
WINDOW_WIDTH = 1300
WINDOW_HEIGHT = 850
SIDEBAR_COLOR = "#1e272e"
SIDEBAR_TEXT_COLOR = "#d2dae2"
ACTIVE_BUTTON_COLOR = "#3c40c6"
BG_COLOR = "#f1f2f6"
ACCENT_COLOR = "#0fb9b1"

class SecurityApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Advanced Security Access Control System")
        self.root.geometry(f"{WINDOW_WIDTH}x{WINDOW_HEIGHT}")
        self.root.configure(bg=BG_COLOR)
        
        # Initialize Backend
        self.db = DatabaseManager()
        self.recognizer = FaceRecognitionEngine(db_manager=self.db)
        
        # Camera Setup using MultiCameraManager (handles Windows MSMF conflicts)
        self.cam_manager = MultiCameraManager()
        self.caps = self.cam_manager.get_slots()
        self.camera_indices = [slot.index for slot in self.caps]

        self.current_frames = [None] * len(self.camera_indices)
        self.is_running = True
        
        # State
        self.active_page = None
        self.active_cam_index = 0
        self.monitor_active = False
        self.selected_room_id = None
        self.monitor_room_config = None # Will hold {allowed_ranks: [...], name: "..."}
        
        # Cache for DB objects (Performance Optimization)
        self.users_cache = []
        self.rooms_cache = []
        self.refresh_db_cache()

        # Recognition Worker
        self.rec_worker = RecognitionWorker(self.recognizer)
        
        # Multi-Camera State tracking
        # Zone names for each camera: (Left, Center, Right)
        # User requested for Cam 1 (idx 1): BLOCK 1, JUNCTION, BLOCK 2
        self.cam_zone_names = [
            ["Room 1", "Hallway", "Room 2"], # Cam 0
            ["BLOCK 1", "JUNCTION", "BLOCK 2"] # Cam 1
        ]
        
        self.detectors = [RoomDetector(max_disappeared=20, zone_names=self.cam_zone_names[i]) for i in range(len(self.camera_indices))]
        self.tracked_names_list = [{} for _ in self.camera_indices] # List of dicts: ID -> Name
        self.pending_unknown_logs_list = [{} for _ in self.camera_indices] # List of dicts: ID -> {id, time}
        self.frame_counts = [0] * len(self.camera_indices)
        
        # Threads
        self.lock = threading.Lock()
        
        # UI Setup
        self.setup_styles()
        self.create_layout()
        
        # Start Camera Loop
        self.update_camera()
        
        # Start with Dashboard
        self.show_dashboard()

    def refresh_db_cache(self):
        """Update local caches for users and rooms to avoid frequent DB queries."""
        # Ensure new room names exist in DB (if DB was already created)
        rooms_to_ensure = ["BLOCK 1", "BLOCK 2", "JUNCTION"]
        existing_rooms = [r['room_name'] for r in self.db.get_rooms()]
        for rname in rooms_to_ensure:
            if rname not in existing_rooms:
                self.db.get_connection().execute('INSERT INTO rooms (room_name, allowed_ranks) VALUES (?, ?)', (rname, "1,2,3"))
                self.db.get_connection().commit()
        
        self.users_cache = self.db.get_users()
        self.rooms_cache = self.db.get_rooms()
        print(f"[System] DB Cache Refreshed. Users: {len(self.users_cache)}, Rooms: {len(self.rooms_cache)}")

    def get_cached_user(self, name):
        return next((u for u in self.users_cache if u['name'] == name), None)

    def get_cached_room(self, room_name):
        return next((r for r in self.rooms_cache if r['room_name'] == room_name), None)

    # ... (Styles and Setup methods remain same as before until show_logs)

    def show_logs(self):
        self.clear_content()
        self.set_active_nav("Access Logs")
        self.active_page = None # Stop video processing
        
        ttk.Label(self.content_frame, text="Security Access Logs", style="Header.TLabel").pack(anchor="w", pady=(0, 20))
        
        # Controls
        ctrl_frame = tk.Frame(self.content_frame, bg=BG_COLOR)
        ctrl_frame.pack(fill="x", pady=(0, 10))
        
        btn_clear = tk.Button(ctrl_frame, text="CLEAR ALL LOGS", bg="#eb4d4b", fg="white", 
                              font=("Segoe UI", 10, "bold"), command=self.clear_all_logs)
        btn_clear.pack(side="right")
        
        cols = ("Time", "Name", "Rank", "Room", "Status", "Confidence")
        self.log_tree = ttk.Treeview(self.content_frame, columns=cols, show="headings", height=20)
        
        for col in cols:
            self.log_tree.heading(col, text=col)
            width = 100
            if col == "Time": width = 150
            if col == "Name": width = 200
            self.log_tree.column(col, width=width)
            
        scrollbar = ttk.Scrollbar(self.content_frame, orient="vertical", command=self.log_tree.yview)
        self.log_tree.configure(yscroll=scrollbar.set)
        
        self.log_tree.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")
        
        self.refresh_logs()

    def refresh_logs(self):
        for item in self.log_tree.get_children():
            self.log_tree.delete(item)
            
        logs = self.db.get_logs(limit=100)
        users = self.db.get_users()
        user_map = {u['name']: u['authority_rank'] for u in users}
        
        for log in logs:
            name = log['person_name']
            rank = str(user_map.get(name, "Unknown/No Rank")) if name != "Unknown" else "--"
            conf = f"{log['confidence']:.2f}" if log['confidence'] else ""
            self.log_tree.insert("", "end", values=(log['timestamp'], name, rank, log['room_name'], log['status'], conf))

    def clear_all_logs(self):
        if messagebox.askyesno("Confirm Clear", "Are you sure you want to delete ALL access logs? This cannot be undone."):
            self.db.clear_logs()
            self.refresh_logs()
            messagebox.showinfo("Success", "Logs cleared.")

    def setup_styles(self):
        self.style = ttk.Style()
        self.style.theme_use('clam')
        
        # Configure common styles
        self.style.configure("TFrame", background=BG_COLOR)
        self.style.configure("TLabel", background=BG_COLOR, font=("Segoe UI", 11))
        self.style.configure("TButton", font=("Segoe UI", 10, "bold"), padding=6)
        
        self.style.configure("Card.TFrame", background="white", relief="flat")
        self.style.configure("Header.TLabel", font=("Segoe UI", 24, "bold"), foreground="#2f3542", background=BG_COLOR)
        self.style.configure("SubHeader.TLabel", font=("Segoe UI", 14, "bold"), foreground="#57606f", background="white")

    def create_layout(self):
        # Sidebar
        self.sidebar = tk.Frame(self.root, bg=SIDEBAR_COLOR, width=250)
        self.sidebar.pack(side="left", fill="y")
        self.sidebar.pack_propagate(False)
        
        # Logo / Title in Sidebar
        tk.Label(self.sidebar, text="SECURE\nACCESS", bg=SIDEBAR_COLOR, fg=ACCENT_COLOR, 
                 font=("Montserrat", 20, "bold"), pady=30).pack(fill="x")
        
        # Navigation Buttons
        self.nav_buttons = {}
        nav_items = [
            ("Dashboard", self.show_dashboard),
            ("New Registration", self.show_registration),
            ("Registered Users", self.show_users),
            ("Room Access Control", self.show_rooms),
            ("Live Monitoring", self.show_monitor),
            ("Access Logs", self.show_logs)
        ]
        
        for name, cmd in nav_items:
            btn = tk.Button(self.sidebar, text=name, bg=SIDEBAR_COLOR, fg=SIDEBAR_TEXT_COLOR,
                            font=("Segoe UI", 11), bd=0, activebackground=ACTIVE_BUTTON_COLOR,
                            activeforeground="white", command=cmd, anchor="w", padx=20, pady=10)
            btn.pack(fill="x")
            self.nav_buttons[name] = btn
            
        # Main Content Area
        self.content_frame = tk.Frame(self.root, bg=BG_COLOR)
        self.content_frame.pack(side="right", fill="both", expand=True, padx=20, pady=20)

    def set_active_nav(self, name):
        # Reset all
        for btn_name, btn in self.nav_buttons.items():
            btn.configure(bg=SIDEBAR_COLOR, font=("Segoe UI", 11))
        # Highlight current
        if name in self.nav_buttons:
            self.nav_buttons[name].configure(bg=ACTIVE_BUTTON_COLOR, font=("Segoe UI", 11, "bold"))

    def clear_content(self):
        self.monitor_active = False # Stop monitoring logic if leaving monitor page
        for widget in self.content_frame.winfo_children():
            widget.destroy()

    # --- Video Loop ---
    def update_camera(self):
        if self.is_running:
            for i, cam in enumerate(self.caps):
                # CameraStream.read() is non-blocking - returns latest frame from thread
                ret, frame = cam.read()
                if ret and frame is not None:
                    self.current_frames[i] = frame  # Already flipped inside CameraStream
            
            # Process all camera frames for monitoring (simultaneous background detection)
            if self.monitor_active:
                # Handle Recognition Worker results FIRST
                rec_results = self.rec_worker.get_results()
                for cam_idx, oid, name, conf, context in rec_results:
                    if name != "Unknown":
                        self.tracked_names_list[cam_idx][oid] = name
                        # Process pending logs for this new identification
                        self._handle_late_identification(cam_idx, oid, name, context)

                for i in range(len(self.caps)):
                    if self.current_frames[i] is not None:
                        self.process_monitor_logics(i, self.current_frames[i])

            # Show registration feed from laptop cam (index 0)
            if self.active_page == "registration" and self.current_frames[0] is not None:
                self.process_registration_frame(self.current_frames[0])

        self.root.after(30, self.update_camera)

    # --- Pages ---

    def show_dashboard(self):
        self.clear_content()
        self.set_active_nav("Dashboard")
        self.active_page = None
        
        ttk.Label(self.content_frame, text="Security Dashboard", style="Header.TLabel").pack(anchor="w", pady=(0, 20))
        
        stats_frame = tk.Frame(self.content_frame, bg=BG_COLOR)
        stats_frame.pack(fill="x")
        
        # Fetch stats
        users = self.db.get_users()
        logs = self.db.get_logs(limit=100)
        unauth_count = sum(1 for l in logs if l['status'] == 'UNAUTHORIZED')
        
        self.create_stat_card(stats_frame, "Total Personnel", len(users), "#4834d4")
        self.create_stat_card(stats_frame, "Unauthorized Events", unauth_count, "#eb4d4b")
        self.create_stat_card(stats_frame, "Total Logs", len(logs), "#6ab04c")
        
        # System Status
        status_frame = tk.Frame(self.content_frame, bg="white", padx=20, pady=20)
        status_frame.pack(fill="x", pady=20)
        
        tk.Label(status_frame, text="System Status", font=("Segoe UI", 14, "bold"), bg="white").pack(anchor="w")
        tk.Label(status_frame, text=f"Database: Connected ({self.db.db_name})", bg="white", fg="green").pack(anchor="w", pady=5)
        
        # Camera Status
        for i, cap in enumerate(self.caps):
            name = "Laptop Cam" if i == 0 else f"External Cam {i}"
            status = "Active" if cap and cap.isOpened() else "Disconnected"
            color = "green" if status == "Active" else "red"
            tk.Label(status_frame, text=f"{name}: {status}", bg="white", fg=color).pack(anchor="w")

    def create_stat_card(self, parent, title, value, color):
        card = tk.Frame(parent, bg="white", width=250, height=120, padx=20, pady=20)
        card.pack(side="left", padx=(0, 20))
        card.pack_propagate(False)
        
        tk.Label(card, text=str(value), font=("Segoe UI", 28, "bold"), fg=color, bg="white").pack(anchor="w")
        tk.Label(card, text=title, font=("Segoe UI", 10), fg="#a4b0be", bg="white").pack(anchor="w")

    def show_registration(self):
        self.clear_content()
        self.set_active_nav("New Registration")
        
        ttk.Label(self.content_frame, text="New Personnel Registration", style="Header.TLabel").pack(anchor="w", pady=(0, 20))
        
        main_container = tk.Frame(self.content_frame, bg=BG_COLOR)
        main_container.pack(fill="both", expand=True)
        
        # --- LEFT: Form Details ---
        form_frame = tk.Frame(main_container, bg="white", padx=20, pady=20)
        form_frame.pack(side="left", fill="both", expand=True, padx=(0, 10))
        
        ttk.Label(form_frame, text="1. Enter Details", style="SubHeader.TLabel").pack(anchor="w", pady=(0, 20))
        
        self.entries = {}
        fields = [("Full Name", "text"), ("Age", "number"), ("Gender", "combo", ["Male", "Female", "Other"]),
                  ("Department", "text"), ("Batch No", "text"), ("Authority Rank", "combo", ["1", "2", "3"])]
        
        for label, ftype, *opts in fields:
            tk.Label(form_frame, text=label, bg="white", fg="#57606f").pack(anchor="w", pady=(10, 2))
            if ftype == "combo":
                ent = ttk.Combobox(form_frame, values=opts[0], state="readonly")
                ent.current(0)
            else:
                ent = ttk.Entry(form_frame)
            ent.pack(fill="x")
            self.entries[label] = ent

        # --- RIGHT: Camera Source ---
        cam_frame = tk.LabelFrame(main_container, text="2. Capture Face", bg=BG_COLOR, font=("Segoe UI", 10, "bold"))
        cam_frame.pack(side="right", fill="both", expand=True, padx=(10, 0), ipadx=10, ipady=10)
        
        self.reg_canvas = tk.Canvas(cam_frame, bg="black", width=300, height=250)
        self.reg_canvas.pack(fill="both", expand=True, padx=5, pady=5)
        
        btn_capture = tk.Button(cam_frame, text="CAPTURE FACE", bg=ACCENT_COLOR, fg="white", 
                                font=("Segoe UI", 11, "bold"), command=self.capture_face)
        btn_capture.pack(fill="x", pady=10)
        
        self.captured_image_label = tk.Label(cam_frame, text="No Image Captured", bg="white")
        self.captured_image_label.pack(fill="x", pady=5, ipady=10)

        # Submit Button (Bottom of Form)
        tk.Button(form_frame, text="REGISTER PERSONNEL", bg="#2ecc71", fg="white", font=("Segoe UI", 12, "bold"),
                  command=self.submit_registration, pady=10).pack(fill="x", pady=30, side="bottom")
        
        # Set active page processing
        self.active_page = "registration"
        self.current_capture_path = None
        self.current_capture_frame = None



    def process_registration_frame(self, frame):
        # Resize to fit canvas
        if hasattr(self, 'reg_canvas'):
            self.display_image(frame, self.reg_canvas)

    def display_image(self, cv_img, canvas):
        # Get canvas dimensions
        w = canvas.winfo_width()
        h = canvas.winfo_height()
        if w < 10 or h < 10: return # Too small/not rendered yet
        
        # Resize maintaining aspect ratio
        img_h, img_w = cv_img.shape[:2]
        scale = min(w/img_w, h/img_h)
        new_w, new_h = int(img_w*scale), int(img_h*scale)
        
        resized = cv2.resize(cv_img, (new_w, new_h))
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(rgb)
        tk_img = ImageTk.PhotoImage(image=pil_img)
        
        canvas.create_image(w//2, h//2, image=tk_img, anchor="center")
        canvas.image = tk_img # Keep reference

    def capture_face(self):
        if self.current_frames[0] is None: return
        
        # Save temp file
        filename = "temp_capture.jpg"
        cv2.imwrite(filename, self.current_frames[0])
        
        # Verify face detection
        embedding = self.recognizer.get_embedding(filename)
        if embedding is None:
            messagebox.showerror("Capture Failed", "No face detected! Please look at the camera.")
            return
            
        self.current_capture_frame = self.current_frames[0].copy()
        
        # Show thumbnail
        thumb = cv2.resize(self.current_capture_frame, (150, 150))
        rgb = cv2.cvtColor(thumb, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(rgb)
        tk_img = ImageTk.PhotoImage(image=pil_img)
        self.captured_image_label.configure(image=tk_img, text="")
        self.captured_image_label.image = tk_img
        
        self.current_capture_path = filename
        messagebox.showinfo("Success", "Face captured successfully!")

    def submit_registration(self):
        if not self.current_capture_path:
            messagebox.showwarning("Incomplete", "Please capture a face first.")
            return
            
        data = {k: v.get() for k, v in self.entries.items()}
        if not data["Full Name"]:
            messagebox.showwarning("Incomplete", "Name is required.")
            return

        # Save Image Permanently
        user_filename = f"{data['Full Name'].replace(' ', '_')}_{int(time.time())}.jpg"
        final_path = os.path.join("users_db", user_filename)
        if not os.path.exists("users_db"): os.makedirs("users_db")
        
        cv2.imwrite(final_path, self.current_capture_frame)
        
        # Get embedding again (or cache it, but re-getting is safer for logic flow simplicity)
        embedding = self.recognizer.get_embedding(final_path)
        
        if embedding:
            self.db.add_user(
                data["Full Name"], 
                data["Age"], 
                data["Gender"], 
                data["Department"], 
                data["Batch No"], 
                data["Authority Rank"], 
                final_path, 
                embedding
            )
            self.refresh_db_cache() # Update cache
            self.recognizer.reload_embeddings()
            messagebox.showinfo("Success", f"{data['Full Name']} Registered Successfully!")
            self.show_users() # Go to users list
        else:
            messagebox.showerror("Error", "Failed to process face embedding.")

    def show_users(self):
        self.clear_content()
        self.set_active_nav("Registered Users")
        self.active_page = None
        
        ttk.Label(self.content_frame, text="Registered Personnel", style="Header.TLabel").pack(anchor="w", pady=(0, 20))
        
        # Controls
        ctrl_frame = tk.Frame(self.content_frame, bg=BG_COLOR)
        ctrl_frame.pack(fill="x", pady=(0, 20))
        
        btn_refresh = tk.Button(ctrl_frame, text="Refresh List", command=self.refresh_users_list)
        btn_refresh.pack(side="left")
        
        btn_del = tk.Button(ctrl_frame, text="Delete Selected", bg="#eb4d4b", fg="white", command=self.delete_selected_user)
        btn_del.pack(side="right")
        
        # Treeview
        cols = ("ID", "Name", "Department", "Rank", "Batch")
        self.user_tree = ttk.Treeview(self.content_frame, columns=cols, show="headings", height=15)
        
        for col in cols:
            self.user_tree.heading(col, text=col)
            self.user_tree.column(col, width=150 if col != "ID" else 50)
            
        self.user_tree.pack(fill="both", expand=True)
        self.refresh_users_list()

    def refresh_users_list(self):
        for item in self.user_tree.get_children():
            self.user_tree.delete(item)
        
        users = self.db.get_users()
        for u in users:
            self.user_tree.insert("", "end", values=(u['id'], u['name'], u['department'], u['authority_rank'], u['batch_no']))

    def delete_selected_user(self):
        selected = self.user_tree.selection()
        if not selected:
            messagebox.showwarning("Warning", "No user selected.")
            return
            
        item = self.user_tree.item(selected[0])
        user_id = item['values'][0]
        name = item['values'][1]
        
        if messagebox.askyesno("Confirm", f"Delete user {name}?"):
            self.db.delete_user(user_id)
            self.refresh_db_cache() # Update cache
            self.recognizer.reload_embeddings()
            self.refresh_users_list()

    def show_rooms(self):
        self.clear_content()
        self.set_active_nav("Room Access Control")
        
        ttk.Label(self.content_frame, text="Room Access Configuration", style="Header.TLabel").pack(anchor="w", pady=(0, 20))
        
        rooms = self.db.get_rooms()
        
        for room in rooms:
            frame = tk.Frame(self.content_frame, bg="white", padx=20, pady=20)
            frame.pack(fill="x", pady=10)
            
            tk.Label(frame, text=room['room_name'], font=("Segoe UI", 16, "bold"), bg="white").pack(side="left")
            
            # Checkboxes for ranks
            allowed = room.get('allowed_ranks', '').split(',')
            
            vars = []
            for rank in ['1', '2', '3']:
                var = tk.BooleanVar(value=(rank in allowed))
                vars.append((rank, var))
                chk = tk.Checkbutton(frame, text=f"Allow Rank {rank}", variable=var, bg="white", 
                                     font=("Segoe UI", 11),
                                     command=lambda r=room['id'], v=vars: self.save_room_config(r, v))
                chk.pack(side="right", padx=10)

    def save_room_config(self, room_id, vars_list):
        # vars_list is list of (rank_str, BooleanVar)
        new_ranks = [rank for rank, var in vars_list if var.get()]
        ranks_str = ",".join(new_ranks)
        self.db.update_room_ranks(room_id, ranks_str)
        self.refresh_db_cache() # Update cache
        # print(f"Output: Updated Room {room_id} with ranks {ranks_str}")

    # Implement monitoring logic properly - MULTI-CAMERA MODEL
    def show_monitor(self):
        self.clear_content()
        self.set_active_nav("Live Monitoring")
        self.active_page = "monitor"
        self.monitor_active = True 

        title_frame = tk.Frame(self.content_frame, bg=BG_COLOR)
        title_frame.pack(fill="x", pady=(0, 10))
        
        ttk.Label(title_frame, text="Live Surveillance - Room Entry Monitor", style="Header.TLabel").pack(side="left")
        
        # Camera Switching Controls
        switch_frame = tk.Frame(title_frame, bg=BG_COLOR)
        switch_frame.pack(side="right")
        
        tk.Button(switch_frame, text="< PREV CAM", bg=SIDEBAR_COLOR, fg="white", 
                  font=("Segoe UI", 10, "bold"), command=self.prev_camera).pack(side="left", padx=5)
        
        self.cam_label = tk.Label(switch_frame, text=f"CAMERA {self.active_cam_index} (Active)", 
                                 bg=BG_COLOR, font=("Segoe UI", 10, "bold"))
        self.cam_label.pack(side="left", padx=10)
        
        tk.Button(switch_frame, text="NEXT CAM >", bg=SIDEBAR_COLOR, fg="white", 
                  font=("Segoe UI", 10, "bold"), command=self.next_camera).pack(side="left", padx=5)

        # Info
        info_frame = tk.Frame(self.content_frame, bg=BG_COLOR)
        info_frame.pack(fill="x", pady=(0, 10))
        
        tk.Label(info_frame, text="Room 1 (Left) | Hallway (Center) | Room 2 (Right)", font=("Segoe UI", 12, "bold"), bg=BG_COLOR).pack(side="left")
        
        # Video Area
        video_frame = tk.Frame(self.content_frame, bg=BG_COLOR)
        video_frame.pack(fill="both", expand=True)
        
        self.monitor_canvas = tk.Canvas(video_frame, bg="black")
        self.monitor_canvas.pack(fill="both", expand=True)
        
        # Initialize tracking state if not already handled in __init__
        # (Already handled in __init__ now)

    def prev_camera(self):
        self.active_cam_index = (self.active_cam_index - 1) % len(self.caps)
        self.cam_label.config(text=f"CAMERA {self.active_cam_index} (Active)")

    def next_camera(self):
        self.active_cam_index = (self.active_cam_index + 1) % len(self.caps)
        self.cam_label.config(text=f"CAMERA {self.active_cam_index} (Active)")

    def _handle_late_identification(self, idx, objectID, name, context):
        """Updates pending unauthorized logs when a person is identified late."""
        if objectID in self.pending_unknown_logs_list[idx]:
            pend = self.pending_unknown_logs_list[idx][objectID]
            # Only update if it happened recently (last 10 seconds)
            if (time.time() - pend['time']) < 10.0 and pend['id']:
                new_status = "AUTHORIZED"
                target_room = self.get_cached_room(pend['room'])
                if target_room:
                    allowed_ranks = str(target_room['allowed_ranks']).split(',')
                    user = self.get_cached_user(name)
                    if not user or str(user['authority_rank']) not in allowed_ranks:
                        new_status = "UNAUTHORIZED"
                
                self.db.update_log_person(pend['id'], None, name, new_status)
                print(f"[System] UPDATE LOG [Cam {idx}]: {pend['id']} -> {name} ({new_status})")
                del self.pending_unknown_logs_list[idx][objectID]

    def process_monitor_logics(self, idx, frame):
        """Processes security logic for a specific camera frame."""
        self.frame_counts[idx] += 1
        
        # --- 1. Detect Faces (Haar) ---
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self.recognizer.face_cascade.detectMultiScale(gray, 1.1, 4)
        
        # --- 2. Update Tracker ---
        FRAME_WIDTH = frame.shape[1]
        FRAME_HEIGHT = frame.shape[0]
        events = self.detectors[idx].update(faces, FRAME_WIDTH)
        current_objects = self.detectors[idx].get_objects()
        
        # --- 3. Handle Events ---
        for (event_type, objectID, location, data) in events:
            if event_type == "ENTERED_ROOM":
                name = self.tracked_names_list[idx].get(objectID, f"Person-{objectID}")
                status = "AUTHORIZED"
                target_room = self.get_cached_room(location)
                
                if target_room:
                    allowed_ranks = str(target_room['allowed_ranks']).split(',')
                    if name.startswith("Person-") or name == "Unknown":
                        status = "UNAUTHORIZED"
                    else:
                        user = self.get_cached_user(name)
                        if not user or str(user['authority_rank']) not in allowed_ranks:
                            status = "UNAUTHORIZED"
                
                log_id = self.db.log_access(None, name, location, status)
                print(f"[System] LOG [Cam {idx}]: {name} entered {location} ({status})")
                
                if status == "UNAUTHORIZED":
                     self.show_security_alert(f"UNAUTHORIZED ACCESS - CAM {idx}\n{location}: {name}")

                if name.startswith("Person-") or name == "Unknown":
                    self.pending_unknown_logs_list[idx][objectID] = {'id': log_id, 'time': time.time(), 'room': location}
                
                if objectID in self.tracked_names_list[idx]:
                    del self.tracked_names_list[idx][objectID]

            elif event_type == "EXITED_ROOM":
                rect_idx = data if data is not None else -1
                if rect_idx >= 0 and rect_idx < len(faces):
                    (x, y, w, h) = faces[rect_idx]
                    if objectID not in self.tracked_names_list[idx] or self.tracked_names_list[idx][objectID].startswith("Person-"):
                        face_img = frame[y:y+h, x:x+w].copy()
                        self.rec_worker.submit(idx, objectID, face_img)
                
                if objectID not in self.tracked_names_list[idx]:
                    self.tracked_names_list[idx][objectID] = f"Person-{objectID}"
            
            elif event_type == "ZONE_CHANGE":
                name = self.tracked_names_list[idx].get(objectID, f"Person-{objectID}")
                center_zone = self.detectors[idx].zone_names[1]
                
                # If entering the junction/hallway, log it as a formal entry if requested
                if location == center_zone:
                    # For Cam 1 (index 1), we treat junction entry as a formal access event
                    status = "AUTHORIZED" if not (name.startswith("Person-") or name == "Unknown") else "UNAUTHORIZED"
                    log_id = self.db.log_access(None, name, location, status)
                    print(f"[System] LOG [Cam {idx}]: {name} entered junction {location} ({status})")
                    
                    if name.startswith("Person-") or name == "Unknown":
                        self.pending_unknown_logs_list[idx][objectID] = {'id': log_id, 'time': time.time(), 'room': location}

        # --- 4. Face Recognition (Offloaded to Worker) ---
        freq = 15 if idx == self.active_cam_index else 45
        if self.frame_counts[idx] % freq == 0:
            for (x, y, w, h) in faces:
                if w < 50 or h < 50: continue
                cX, cY = int(x + w/2), int(y + h/2)
                best_oid = None
                min_dist = 100
                for oid, (objX, objY) in current_objects.items():
                    if np.linalg.norm(np.array([cX, cY]) - np.array([objX, objY])) < min_dist:
                        best_oid = oid; min_dist = np.linalg.norm(np.array([cX, cY]) - np.array([objX, objY]))
                        
                if best_oid is not None:
                    if best_oid not in self.tracked_names_list[idx] or self.tracked_names_list[idx][best_oid].startswith("Person-"):
                        face_img = frame[y:y+h, x:x+w].copy()
                        self.rec_worker.submit(idx, best_oid, face_img)

        # --- 5. Draw Visuals (Only for Active Camera) ---
        if idx == self.active_cam_index and hasattr(self, 'monitor_canvas') and self.monitor_canvas.winfo_exists():
            display_frame = frame.copy()
            margin = int(FRAME_WIDTH * 0.20)
            overlay = display_frame.copy()
            cv2.rectangle(overlay, (0, 0), (margin, FRAME_HEIGHT), (0, 255, 0), -1)
            cv2.rectangle(overlay, (margin, 0), (FRAME_WIDTH-margin, FRAME_HEIGHT), (255, 200, 0), -1)
            cv2.rectangle(overlay, (FRAME_WIDTH-margin, 0), (FRAME_WIDTH, FRAME_HEIGHT), (0, 255, 255), -1)
            cv2.addWeighted(overlay, 0.15, display_frame, 0.85, 0, display_frame)
            
            # Use detector's names for labels
            names = self.detectors[idx].zone_names
            cv2.putText(display_frame, names[0], (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
            cv2.putText(display_frame, names[1], (FRAME_WIDTH//2 - 60, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
            cv2.putText(display_frame, names[2], (FRAME_WIDTH-140, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

            for (x, y, w, h) in faces:
                cX, cY = int(x + w/2), int(y + h/2)
                label = "Unknown"
                color = (0, 0, 255)
                for oid, (objX, objY) in current_objects.items():
                     if np.linalg.norm(np.array([cX, cY]) - np.array([objX, objY])) < 100:
                         if oid in self.tracked_names_list[idx]:
                             label = self.tracked_names_list[idx][oid]
                             if not label.startswith("Person-"): color = (0, 255, 0)
                         else: label = f"ID {oid}"
                
                cv2.rectangle(display_frame, (x, y), (x+w, y+h), color, 2)
                cv2.putText(display_frame, label, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

            self.display_image(display_frame, self.monitor_canvas)

    def show_security_alert(self, message):
        """Shows a non-blocking alert in the UI."""
        print(f"!!! SECURITY ALERT !!! {message}")
        if hasattr(self, 'root'):
            # Create a transient top-level or just a label if we had a dedicated status bar
            # For now, let's use a simple toast-like label if we are on the monitor page
            if self.active_page == "monitor":
                alert_lbl = tk.Label(self.content_frame, text=message, bg="#eb4d4b", fg="white", 
                                    font=("Segoe UI", 12, "bold"), pady=10)
                alert_lbl.pack(fill="x", side="bottom")
                self.root.after(5000, alert_lbl.destroy)

    def refresh_logs(self):
        for item in self.log_tree.get_children():
            self.log_tree.delete(item)
            
        logs = self.db.get_logs(limit=100)
        users = self.db.get_users()
        user_map = {u['name']: u['authority_rank'] for u in users}
        
        for log in logs:
            name = log['person_name']
            rank = str(user_map.get(name, "Unknown/No Rank")) if name != "Unknown" else "--"
            conf = f"{log['confidence']:.2f}" if log['confidence'] else ""
            self.log_tree.insert("", "end", values=(log['timestamp'], name, rank, log['room_name'], log['status'], conf))

    def clear_all_logs(self):
        if messagebox.askyesno("Confirm Clear", "Are you sure you want to delete ALL access logs? This cannot be undone."):
            self.db.clear_logs()
            self.refresh_logs()
            messagebox.showinfo("Success", "Logs cleared.")
    
    def on_closing(self):
        self.is_running = False
        if hasattr(self, 'rec_worker'):
            self.rec_worker.stop()
        if hasattr(self, 'cam_manager'):
            self.cam_manager.release_all()
        else:
            for cam in getattr(self, 'caps', []):
                if cam: cam.release()
        self.root.destroy()


if __name__ == "__main__":
    root = tk.Tk()
    app = SecurityApp(root)
    root.protocol("WM_DELETE_WINDOW", app.on_closing)
    root.mainloop()
