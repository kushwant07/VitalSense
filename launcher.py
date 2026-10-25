import customtkinter as ctk
import os
import sys
import subprocess
import platform
import time
import threading
import gui_app 

ctk.set_appearance_mode("Light")
ctk.set_default_color_theme("blue")

class LauncherApp(ctk.CTk):
    def __init__(self, skip_loading=False):
        super().__init__()
        
        self.title("VitalSense - Medical Kiosk")
        self.configure(fg_color="#f8fafc")
        
        screen_width = self.winfo_screenwidth()
        screen_height = self.winfo_screenheight()
        self.geometry(f"{screen_width}x{screen_height}+0+0")
        
        # Start invisible for fade-in effect
        self.attributes("-alpha", 0.0)
        
        self.grid_columnconfigure(0, weight=1)
        self.grid_rowconfigure(0, weight=1)

        if skip_loading:
            self.show_main_menu()
            self.animate_fade_in()
        else:
            self.show_loading_screen()
            self.animate_fade_in()
            self.after(800, self.run_boot_sequence)

    def animate_fade_in(self):
        """Smoothly fades the window in from 0 to 1 opacity"""
        alpha = 0.0
        while alpha < 1.0:
            alpha += 0.05
            self.attributes("-alpha", alpha)
            self.update()
            time.sleep(0.01)

    def animate_fade_out_and_destroy(self, next_action):
        """Smoothly fades out then executes the next action"""
        alpha = 1.0
        while alpha > 0.0:
            alpha -= 0.05
            self.attributes("-alpha", alpha)
            self.update()
            time.sleep(0.01)
        self.destroy()
        next_action()

    def show_loading_screen(self):
        self.frame_loading = ctk.CTkFrame(self, fg_color="#f8fafc")
        self.frame_loading.grid(row=0, column=0, sticky="nsew")
        
        self.frame_loading.grid_columnconfigure(0, weight=1)
        self.frame_loading.grid_rowconfigure(0, weight=1)
        self.frame_loading.grid_rowconfigure(1, weight=0)
        self.frame_loading.grid_rowconfigure(2, weight=1)

        self.lbl_boot = ctk.CTkLabel(
            self.frame_loading, 
            text="VITALSENSE AI", 
            font=("Roboto", 42, "bold"), 
            text_color="#0284c7"
        )
        self.lbl_boot.grid(row=0, column=0, sticky="s", pady=(0, 20))
        
        self.progress = ctk.CTkProgressBar(
            self.frame_loading, 
            width=600, 
            height=8, 
            progress_color="#0284c7",
            fg_color="#e2e8f0"
        )
        self.progress.grid(row=1, column=0, sticky="n")
        self.progress.set(0)
        
        self.lbl_status = ctk.CTkLabel(
            self.frame_loading, 
            text="Initializing System Hardware...", 
            font=("Roboto Mono", 14), 
            text_color="#64748b"
        )
        self.lbl_status.grid(row=2, column=0, sticky="n", pady=(20, 0))

    def run_boot_sequence(self):
        steps = [
            ("Initializing Camera Hardware...", 0.3),
            ("Mounting Diagnostic Storage...", 0.5),
            ("Loading Deep Learning Models...", 0.7),
            ("Preparing Medical Interface...", 0.9),
            ("System Ready.", 1.0)
        ]
        
        def _boot():
            current_prog = 0.0
            for text, target in steps:
                self.lbl_status.configure(text=text)
                while current_prog < target:
                    current_prog += 0.01
                    self.progress.set(current_prog)
                    time.sleep(0.02)
                time.sleep(0.2)

            time.sleep(0.5)
            self.show_main_menu()

        threading.Thread(target=_boot, daemon=True).start()

    def show_main_menu(self):
        if hasattr(self, 'frame_loading'):
            self.frame_loading.destroy()
        
        self.grid_columnconfigure(0, weight=1)
        self.grid_rowconfigure(0, weight=1)
        self.grid_rowconfigure(1, weight=0)
        self.grid_rowconfigure(2, weight=1)

        self.center_frame = ctk.CTkFrame(self, fg_color="transparent")
        self.center_frame.grid(row=1, column=0, sticky="ns")

        self.lbl_title = ctk.CTkLabel(
            self.center_frame, 
            text="VitalSense AI", 
            font=("Roboto Medium", 68, "bold"), 
            text_color="#0f172a"
        )
        self.lbl_title.pack(pady=(0, 10))
        
        self.lbl_subtitle = ctk.CTkLabel(
            self.center_frame, 
            text="Contactless Clinical Vital Signs Monitoring", 
            font=("Roboto", 22), 
            text_color="#64748b"
        )
        self.lbl_subtitle.pack(pady=(0, 50))

        btn_font = ("Roboto Medium", 18, "bold")
        btn_w, btn_h = 360, 65

        self.btn_start = ctk.CTkButton(
            self.center_frame, 
            text="🩺  START DIAGNOSIS", 
            font=btn_font, 
            height=btn_h, 
            width=btn_w, 
            fg_color="#0284c7", 
            hover_color="#0369a1", 
            text_color="white",
            corner_radius=12, 
            command=self.launch_scanner
        )
        self.btn_start.pack(pady=12)

        self.btn_outputs = ctk.CTkButton(
            self.center_frame, 
            text="📊  VIEW REPORT HISTORY", 
            font=btn_font, 
            height=btn_h, 
            width=btn_w, 
            fg_color="#059669", 
            hover_color="#047857", 
            text_color="white",
            corner_radius=12, 
            command=self.open_outputs_folder
        )
        self.btn_outputs.pack(pady=12)

        self.btn_exit = ctk.CTkButton(
            self.center_frame, 
            text="❌  EXIT APPLICATION", 
            font=btn_font, 
            height=btn_h, 
            width=btn_w, 
            fg_color="#e2e8f0", 
            hover_color="#cbd5e1", 
            text_color="#334155",
            corner_radius=12, 
            command=self.exit_app
        )
        self.btn_exit.pack(pady=(30, 0))

        self.lbl_footer = ctk.CTkLabel(
            self, 
            text="v2.0 Production Build | Cross-Platform Medical Research Suite", 
            font=("Roboto", 12), 
            text_color="#94a3b8"
        )
        self.lbl_footer.grid(row=2, column=0, pady=20, sticky="s")

    def launch_scanner(self):
        from patient_dialog import PatientDetailsDialog
        PatientDetailsDialog(self, on_submit=self._on_patient_submitted)

    def _on_patient_submitted(self, patient_info):
        self.patient_info = patient_info
        self.animate_fade_out_and_destroy(self._open_gui)

    def _open_gui(self):
        try:
            patient_info = getattr(self, 'patient_info', None)
            app = gui_app.VitalSenseApp(patient_info=patient_info)
            w, h = app.winfo_screenwidth(), app.winfo_screenheight()
            app.geometry(f"{w}x{h}+0+0")
            app.mainloop()
        except Exception as e:
            print(f"Failed to launch app: {e}")

    def open_outputs_folder(self):
        path = os.path.abspath("outputs")
        if not os.path.exists(path): 
            os.makedirs(path)
        system_name = platform.system()
        try:
            if system_name == "Windows": 
                os.startfile(path)
            elif system_name == "Darwin": 
                subprocess.Popen(["open", path])
            else: 
                subprocess.Popen(["xdg-open", path])
        except Exception as e: 
            print(f"Error opening outputs folder: {e}")

    def exit_app(self):
        self.animate_fade_out_and_destroy(sys.exit)

if __name__ == "__main__":
    app = LauncherApp()
    app.mainloop()
