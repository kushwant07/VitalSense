import platform
import customtkinter as ctk
from PIL import Image, ImageTk
import cv2
import time
import threading
import numpy as np
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure

# Backend Imports
from vital_monitor import ProductionVitalMonitor
import config
from patient_dialog import PatientDetailsDialog

ctk.set_appearance_mode("Light")
ctk.set_default_color_theme("blue")

def open_webcam(index=0):
    """Robust cross-platform webcam opener for macOS and Windows."""
    if platform.system() == "Windows":
        try:
            cap = cv2.VideoCapture(index, cv2.CAP_DSHOW)
            if cap is not None and cap.isOpened():
                return cap
        except Exception:
            pass
        try:
            cap = cv2.VideoCapture(index, cv2.CAP_MSMF)
            if cap is not None and cap.isOpened():
                return cap
        except Exception:
            pass
    elif platform.system() == "Darwin":
        try:
            cap = cv2.VideoCapture(index, cv2.CAP_AVFOUNDATION)
            if cap is not None and cap.isOpened():
                return cap
        except Exception:
            pass
    return cv2.VideoCapture(index)

class CameraWorker:
    def __init__(self, monitor_instance):
        self.monitor = monitor_instance
        self.cap = None
        self.running = False
        self.recording = False
        self.previewing = False
        
        self.buffer = []
        self.latest_frame = None
        self.lock = threading.Lock()
        
        self.start_time = 0
        self.face_detected = False
        self.frame_count = 0

    def start(self):
        self.running = True
        self.previewing = True
        if self.cap is None or not self.cap.isOpened():
            self.cap = open_webcam(0)
            if self.cap and self.cap.isOpened():
                self.cap.set(cv2.CAP_PROP_FPS, 30)
                self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        self.thread = threading.Thread(target=self._run_loop, daemon=True)
        self.thread.start()

    def stop(self):
        self.running = False
        self.recording = False
        self.previewing = False
        if hasattr(self, 'thread') and self.thread.is_alive():
            self.thread.join(timeout=1.0)
        if self.cap:
            self.cap.release()

    def start_recording(self):
        with self.lock:
            self.buffer = []
            self.frame_count = 0
            self.start_time = time.time()
            self.recording = True

    def stop_recording(self):
        with self.lock:
            self.recording = False

    def pause_preview(self):
        self.previewing = False

    def resume_preview(self):
        self.previewing = True

    def get_latest_frame(self):
        with self.lock:
            return self.latest_frame, self.face_detected

    def _run_loop(self):
        if self.cap is None or not self.cap.isOpened():
            self.cap = open_webcam(0)
            if self.cap and self.cap.isOpened():
                self.cap.set(cv2.CAP_PROP_FPS, 30)
                self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        
        frame_interval = 1.0 / 30.0
        
        while self.running:
            if not self.previewing:
                time.sleep(0.05)
                continue

            if self.cap and self.cap.isOpened():
                ret, frame = self.cap.read()
                if not ret: 
                    time.sleep(0.01)
                    continue

                frame = cv2.flip(frame, 1)
                
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                res = self.monitor.face_mesh.process(rgb)
                
                roi = None
                face_found = False
                if res and res.multi_face_landmarks:
                    face_found = True
                    if self.recording:
                        roi = self.monitor.extract_forehead_roi(frame, res.multi_face_landmarks[0])
                elif self.recording:
                    # Fallback to last known forehead ROI so frames are never dropped during blinks
                    roi = self.monitor.get_last_roi(frame)

                with self.lock:
                    self.latest_frame = frame
                    self.face_detected = face_found
                    
                    if self.recording and roi is not None:
                        self.buffer.append({'fh': roi, 'time': time.time()})
                        if len(self.buffer) >= config.CAPTURE_FRAMES:
                            self.recording = False

                self.frame_count += 1

        if self.cap: 
            self.cap.release()

class VitalSenseApp(ctk.CTk):
    def __init__(self, patient_info=None):
        super().__init__()
        self.title("VitalSense AI - Clinical Diagnostics Monitor")
        self.geometry("1400x900")
        self.configure(fg_color="#f1f5f9")
        self.protocol("WM_DELETE_WINDOW", self.on_close)
        
        # Patient info
        self.patient_info = patient_info or {"name": "", "age": "", "gender": "", "notes": ""}
        
        # Start invisible for fade-in
        self.attributes("-alpha", 0.0)
        
        self.app_state = "LOADING" 
        self.monitor = None
        self.camera = None
        self.running = True
        
        self.grid_columnconfigure(0, weight=3) 
        self.grid_columnconfigure(1, weight=1) 
        self.grid_rowconfigure(0, weight=1)
        
        self.setup_ui()
        
        self.after(100, self.animate_fade_in)
        self.after(400, self.load_backend)

    def get_display_patient_text(self):
        name = self.patient_info.get("name", "").strip() or "Anonymous Subject"
        extras = []
        if self.patient_info.get("age"):
            extras.append(f"{self.patient_info['age']}y")
        if self.patient_info.get("gender"):
            extras.append(self.patient_info["gender"])
        extra_str = f"  •  {', '.join(extras)}" if extras else ""
        return f"{name}{extra_str}"

    def open_patient_dialog(self, on_success=None):
        def _handle_submit(info):
            self.patient_info = info
            if self.monitor:
                self.monitor.set_patient_info(info)
            if hasattr(self, 'lbl_patient_name'):
                self.lbl_patient_name.configure(text=f"👤  {self.get_display_patient_text()}")
            if on_success:
                on_success()

        PatientDetailsDialog(self, current_info=self.patient_info, on_submit=_handle_submit)

    def animate_fade_in(self):
        alpha = 0.0
        while alpha < 1.0:
            alpha += 0.05
            self.attributes("-alpha", alpha)
            self.update()
            time.sleep(0.01)

    def animate_fade_out(self):
        alpha = 1.0
        while alpha > 0.0:
            alpha -= 0.05
            self.attributes("-alpha", alpha)
            self.update()
            time.sleep(0.01)

    def setup_ui(self):
        # Camera Feed Panel (Left)
        self.cam_frame = ctk.CTkFrame(self, corner_radius=16, fg_color="#ffffff", border_color="#e2e8f0", border_width=1)
        self.cam_frame.grid(row=0, column=0, sticky="nsew", padx=20, pady=20)
        self.cam_frame.grid_rowconfigure(0, weight=1)
        self.cam_frame.grid_columnconfigure(0, weight=1)

        self.cam_label = ctk.CTkLabel(
            self.cam_frame, 
            text="Initializing Camera Feed...", 
            font=("Roboto", 16),
            text_color="#64748b"
        )
        self.cam_label.grid(row=0, column=0, sticky="nsew", padx=10, pady=10)

        self.progress_bar = ctk.CTkProgressBar(
            self.cam_frame, 
            height=14, 
            progress_color="#0284c7", 
            fg_color="#e2e8f0"
        )
        self.progress_bar.grid(row=1, column=0, sticky="ew", padx=20, pady=(0, 20))
        self.progress_bar.set(0)

        # Diagnostics Panel (Right)
        self.data_panel = ctk.CTkFrame(self, corner_radius=16, fg_color="#ffffff", border_color="#e2e8f0", border_width=1)
        self.data_panel.grid(row=0, column=1, sticky="nsew", padx=(0, 20), pady=20)
        self.data_panel.grid_columnconfigure(0, weight=1)
        self.data_panel.grid_columnconfigure(1, weight=1)

        ctk.CTkLabel(
            self.data_panel, 
            text="CLINICAL METRICS", 
            font=("Roboto", 20, "bold"), 
            text_color="#0f172a"
        ).grid(row=0, column=0, columnspan=2, pady=(18, 8))

        # Subject / Patient Badge Card
        self.patient_card = ctk.CTkFrame(
            self.data_panel, 
            fg_color="#f8fafc", 
            corner_radius=10, 
            border_color="#38bdf8", 
            border_width=1
        )
        self.patient_card.grid(row=1, column=0, columnspan=2, sticky="ew", padx=12, pady=(0, 10))
        self.patient_card.grid_columnconfigure(0, weight=1)
        self.patient_card.grid_columnconfigure(1, weight=0)

        self.lbl_patient_name = ctk.CTkLabel(
            self.patient_card, 
            text=f"👤  {self.get_display_patient_text()}", 
            font=("Roboto", 13, "bold"), 
            text_color="#0369a1", 
            anchor="w"
        )
        self.lbl_patient_name.grid(row=0, column=0, sticky="w", padx=12, pady=8)

        self.btn_edit_patient = ctk.CTkButton(
            self.patient_card, 
            text="Edit Details", 
            width=84, 
            height=28, 
            font=("Roboto", 11, "bold"), 
            fg_color="#e2e8f0", 
            hover_color="#cbd5e1", 
            text_color="#0f172a",
            corner_radius=6, 
            command=self.open_patient_dialog
        )
        self.btn_edit_patient.grid(row=0, column=1, sticky="e", padx=8, pady=8)

        # 4 Vital Sign Metric Cards
        self.card_hr = self.create_card("HEART RATE", "--", "BPM", "#ef4444", 2, 0)
        self.card_hrv = self.create_card("HRV (RMSSD)", "--", "ms", "#0284c7", 2, 1)
        self.card_br = self.create_card("BREATHING", "--", "BrPM", "#10b981", 3, 0)
        self.card_stress = self.create_card("STRESS LVL", "--", "%", "#f59e0b", 3, 1)

        # Pulse Waveform Graph
        self.graph_container = ctk.CTkFrame(self.data_panel, fg_color="#f8fafc", corner_radius=10, border_color="#e2e8f0", border_width=1)
        self.graph_container.grid(row=4, column=0, columnspan=2, sticky="nsew", padx=12, pady=12)
        self.data_panel.grid_rowconfigure(4, weight=1)

        self.fig = Figure(figsize=(4, 3), dpi=100, facecolor='#f8fafc')
        self.ax = self.fig.add_subplot(111)
        self.ax.set_facecolor('#ffffff')
        self.ax.tick_params(colors='#64748b', labelbottom=False, labelleft=True, labelsize=8)
        self.ax.spines['bottom'].set_color('#cbd5e1')
        self.ax.spines['left'].set_color('#cbd5e1')
        self.ax.spines['top'].set_visible(False)
        self.ax.spines['right'].set_visible(False)
        self.ax.set_title("Waiting for Session Data...", color='#64748b', fontsize=10)
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.graph_container)
        self.canvas.get_tk_widget().pack(fill="both", expand=True, padx=4, pady=4)

        # Status label
        self.status_lbl = ctk.CTkLabel(
            self.data_panel, 
            text="System Initializing...", 
            font=("Roboto Medium", 14), 
            text_color="#64748b"
        )
        self.status_lbl.grid(row=5, column=0, columnspan=2, pady=(6, 6))

        # Main Action Button
        self.btn_action = ctk.CTkButton(
            self.data_panel, 
            text="START MONITORING", 
            height=48, 
            font=("Roboto", 16, "bold"), 
            fg_color="#0284c7", 
            hover_color="#0369a1",
            text_color="white",
            corner_radius=10,
            command=self.handle_button_click
        )
        self.btn_action.grid(row=6, column=0, columnspan=2, sticky="ew", padx=20, pady=(8, 8))
        self.btn_action.configure(state="disabled")

        # Back Button
        self.btn_back = ctk.CTkButton(
            self.data_panel, 
            text="← BACK TO MENU", 
            height=38, 
            font=("Roboto", 13, "bold"), 
            fg_color="#f1f5f9", 
            hover_color="#e2e8f0", 
            text_color="#334155",
            corner_radius=10,
            command=self.return_to_menu
        )
        self.btn_back.grid(row=7, column=0, columnspan=2, sticky="ew", padx=20, pady=(0, 16))

    def create_card(self, title, val, unit, color, r, c):
        f = ctk.CTkFrame(self.data_panel, fg_color="#f8fafc", border_color=color, border_width=2, corner_radius=10)
        f.grid(row=r, column=c, sticky="ew", padx=5, pady=5)
        ctk.CTkLabel(f, text=title, font=("Roboto", 11, "bold"), text_color="#475569").pack(pady=(8, 0))
        lbl = ctk.CTkLabel(f, text=val, font=("Roboto", 28, "bold"), text_color="#0f172a")
        lbl.pack(pady=0)
        ctk.CTkLabel(f, text=unit, font=("Roboto", 11), text_color="#64748b").pack(pady=(0, 8))
        f.value = lbl
        return f

    def load_backend(self):
        self.status_lbl.configure(text="Loading Deep Learning Models...", text_color="#0284c7")
        self.update()
        try:
            self.monitor = ProductionVitalMonitor(patient_info=self.patient_info)
            self.camera = CameraWorker(self.monitor)
            self.camera.start()
            self.app_state = "IDLE"
            self.status_lbl.configure(text="Ready to Scan", text_color="#10b981")
            self.after(33, self.update_gui_loop)
        except Exception as e:
            self.status_lbl.configure(text=f"Initialization Error: {e}", text_color="#ef4444")
            print(f"Error loading backend: {e}")

    def start_new_camera_session(self):
        self.progress_bar.configure(mode="determinate")
        self.progress_bar.set(0)
        self.camera.start()
        self.status_lbl.configure(text="Ready to Scan", text_color="#10b981")

    def update_gui_loop(self):
        if not self.running:
            return

        try:
            with self.camera.lock:
                frame = self.camera.latest_frame
                face_detected = self.camera.face_detected

            if frame is not None:
                h, w = frame.shape[:2]
                disp_w = self.cam_label.winfo_width()
                disp_h = self.cam_label.winfo_height()
                
                if disp_w > 1 and disp_h > 1:
                    scale = min(disp_w/w, disp_h/h)
                    nw, nh = int(w*scale), int(h*scale)
                    frame_resized = cv2.resize(frame, (nw, nh), interpolation=cv2.INTER_LINEAR)
                    rgb = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB)
                    
                    img = Image.fromarray(rgb)
                    imgtk = ctk.CTkImage(light_image=img, dark_image=img, size=(nw, nh))
                    self.cam_label.configure(image=imgtk, text="")
                    self.cam_label.image = imgtk

            if self.app_state == "IDLE":
                if face_detected:
                    self.status_lbl.configure(text="Face Detected - Ready to Scan", text_color="#10b981")
                    self.btn_action.configure(state="normal", text="START MONITORING", fg_color="#0284c7")
                else:
                    self.status_lbl.configure(text="No Face Detected", text_color="#d97706")
                    self.btn_action.configure(state="disabled")

            elif self.app_state == "RECORDING":
                with self.camera.lock:
                    rec_count = len(self.camera.buffer)
                    is_recording = self.camera.recording
                
                prog = rec_count / config.CAPTURE_FRAMES
                self.progress_bar.set(prog)
                self.status_lbl.configure(text=f"Recording Signal... {rec_count}/{config.CAPTURE_FRAMES}", text_color="#0284c7")
                
                if not is_recording and rec_count >= 10:
                    self.finish_recording()

            self.after(33, self.update_gui_loop)
            
        except Exception:
            pass

    def handle_button_click(self):
        if self.app_state == "IDLE":
            if not self.patient_info.get("name"):
                self.open_patient_dialog(on_success=self.start_recording_flow)
                return
            self.start_recording_flow()
            
        elif self.app_state == "RECORDING":
            self.camera.stop_recording()
            self.finish_recording()
            
        elif self.app_state == "RESULTS":
            self.app_state = "IDLE"
            self.progress_bar.set(0)
            self.ax.clear()
            self.ax.set_facecolor('#ffffff')
            self.ax.tick_params(colors='#64748b', labelbottom=False, labelleft=True, labelsize=8)
            self.ax.spines['bottom'].set_color('#cbd5e1')
            self.ax.spines['left'].set_color('#cbd5e1')
            self.ax.spines['top'].set_visible(False)
            self.ax.spines['right'].set_visible(False)
            self.ax.set_title("Waiting for Session Data...", color='#64748b', fontsize=10)
            self.canvas.draw()
            for card in [self.card_hr, self.card_hrv, self.card_br, self.card_stress]:
                card.value.configure(text="--")
            self.start_new_camera_session()

    def start_recording_flow(self):
        self.app_state = "RECORDING"
        self.btn_action.configure(text="STOP RECORDING", fg_color="#ef4444")
        self.camera.start_recording()

    def finish_recording(self):
        self.app_state = "PROCESSING"
        self.camera.stop()
        
        self.status_lbl.configure(text="Processing Waveforms... Please Wait", text_color="#0284c7")
        self.cam_label.configure(image=None, text="⚙️  PROCESSING PHYSIOLOGICAL DATA  ⚙️", text_color="#0284c7")
        self.progress_bar.configure(mode="indeterminate")
        self.progress_bar.start()
        self.btn_action.configure(state="disabled")
        
        threading.Thread(target=self.run_analysis).start()

    def run_analysis(self):
        with self.camera.lock:
            raw_buffer = self.camera.buffer[:]
        
        valid_frames = len(raw_buffer)
        if valid_frames < 30:
            self.after(0, lambda: self.show_error("Insufficient frames recorded. Please scan again."))
            return

        np_buffer = np.zeros(valid_frames, dtype=config.DTYPE_BUFFER)
        for i, item in enumerate(raw_buffer):
            np_buffer[i]['fh'] = item['fh']
            np_buffer[i]['time'] = item['time']

        try:
            self.monitor.set_patient_info(self.patient_info)
            results = self.monitor.process_full_recording(np_buffer, valid_frames)
            self.after(0, lambda: self.show_results(results))
        except Exception as e:
            err_msg = str(e)
            self.after(0, lambda: self.show_error(err_msg))

    def show_results(self, res):
        self.progress_bar.stop()
        self.progress_bar.configure(mode="determinate")
        self.progress_bar.set(1)
        
        if not res:
            self.show_error("No result returned from diagnostic engine")
            return

        self.card_hr.value.configure(text=f"{res['hr']:.1f}")
        self.card_hrv.value.configure(text=f"{res['hrv']:.1f}")
        self.card_br.value.configure(text=f"{res['br']:.1f}")
        self.card_stress.value.configure(text=f"{res['stress']:.1f}")
        
        self.ax.clear()
        self.ax.set_facecolor('#ffffff')
        self.ax.plot(res['waveform'], color='#0284c7', linewidth=2.0)
        self.ax.set_title("Pulse Waveform (30s)", color='#0f172a', fontsize=11, fontweight='bold')
        self.ax.grid(True, color='#e2e8f0', linestyle=':')
        self.ax.spines['bottom'].set_color('#cbd5e1')
        self.ax.spines['left'].set_color('#cbd5e1')
        self.ax.spines['top'].set_visible(False)
        self.ax.spines['right'].set_visible(False)
        self.ax.tick_params(colors='#64748b', labelbottom=False, labelleft=True, labelsize=8)
        self.canvas.draw()
        
        self.monitor.save_plots(res['waveform'], res['fps'], res['score'])
        
        self.app_state = "RESULTS"
        self.status_lbl.configure(text="Analysis Complete — Report Saved", text_color="#10b981")
        self.btn_action.configure(state="normal", text="START NEW SCAN", fg_color="#059669")

    def show_error(self, msg):
        self.progress_bar.stop()
        self.app_state = "RESULTS"
        self.status_lbl.configure(text=f"Error: {msg}", text_color="#ef4444")
        self.btn_action.configure(state="normal", text="TRY AGAIN", fg_color="#ef4444")

    def return_to_menu(self):
        self.running = False
        if self.camera:
            self.camera.stop()
        
        self.destroy() 
        import launcher
        app = launcher.LauncherApp(skip_loading=True)
        app.mainloop()

    def on_close(self):
        self.running = False
        if self.camera:
            self.camera.stop()
        self.quit()
        self.destroy()
        import sys
        sys.exit()

if __name__ == "__main__":
    app = VitalSenseApp()
    app.mainloop()
