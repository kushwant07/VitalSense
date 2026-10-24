import customtkinter as ctk
import re

class PatientDetailsDialog(ctk.CTkToplevel):
    def __init__(self, parent, current_info=None, on_submit=None):
        super().__init__(parent)
        self.title("VitalSense - Subject Details")
        self.geometry("480x520")
        self.resizable(False, False)
        
        self.on_submit = on_submit
        self.submitted = False
        
        if current_info is None:
            current_info = {}
        self.current_info = current_info

        # Modal configuration
        self.transient(parent)
        self.grab_set()
        
        # Center the dialog on the screen/parent
        self.update_idletasks()
        try:
            pw = parent.winfo_width()
            ph = parent.winfo_height()
            px = parent.winfo_rootx()
            py = parent.winfo_rooty()
            if pw > 100 and ph > 100:
                x = px + (pw - 480) // 2
                y = py + (ph - 520) // 2
            else:
                x = (self.winfo_screenwidth() - 480) // 2
                y = (self.winfo_screenheight() - 520) // 2
            self.geometry(f"480x520+{max(0, x)}+{max(0, y)}")
        except Exception:
            pass

        self._build_ui()

    def _build_ui(self):
        container = ctk.CTkFrame(self, fg_color="#1e1e1e", corner_radius=0)
        container.pack(fill="both", expand=True, padx=20, pady=20)

        # Header
        lbl_icon = ctk.CTkLabel(container, text="🩺", font=("Arial", 36))
        lbl_icon.pack(pady=(15, 5))

        lbl_title = ctk.CTkLabel(container, text="Subject Information", font=("Roboto Medium", 22, "bold"), text_color="#3498db")
        lbl_title.pack(pady=(0, 5))

        lbl_sub = ctk.CTkLabel(container, text="Reports will be saved under this person's name", font=("Roboto", 13), text_color="#888888")
        lbl_sub.pack(pady=(0, 20))

        # Form Frame
        form = ctk.CTkFrame(container, fg_color="transparent")
        form.pack(fill="x", padx=15)

        # Name Field (Required)
        ctk.CTkLabel(form, text="Full Name / Patient ID *", font=("Roboto", 13, "bold"), anchor="w").pack(fill="x", pady=(5, 2))
        self.entry_name = ctk.CTkEntry(form, placeholder_text="e.g. John Doe", height=38, font=("Roboto", 14), corner_radius=8)
        self.entry_name.pack(fill="x", pady=(0, 10))
        if self.current_info.get("name"):
            self.entry_name.insert(0, self.current_info["name"])

        # Age and Gender row
        row_age_gender = ctk.CTkFrame(form, fg_color="transparent")
        row_age_gender.pack(fill="x", pady=(0, 10))
        row_age_gender.grid_columnconfigure(0, weight=1)
        row_age_gender.grid_columnconfigure(1, weight=1)

        # Age
        ctk.CTkLabel(row_age_gender, text="Age (Optional)", font=("Roboto", 13), anchor="w").grid(row=0, column=0, sticky="w", padx=(0, 5), pady=(0, 2))
        self.entry_age = ctk.CTkEntry(row_age_gender, placeholder_text="e.g. 28", height=38, font=("Roboto", 14), corner_radius=8)
        self.entry_age.grid(row=1, column=0, sticky="ew", padx=(0, 5))
        if self.current_info.get("age"):
            self.entry_age.insert(0, str(self.current_info["age"]))

        # Gender
        ctk.CTkLabel(row_age_gender, text="Gender (Optional)", font=("Roboto", 13), anchor="w").grid(row=0, column=1, sticky="w", padx=(5, 0), pady=(0, 2))
        gender_options = ["Select Gender", "Male", "Female", "Other", "Prefer not to say"]
        default_gender = self.current_info.get("gender") if self.current_info.get("gender") in gender_options else "Select Gender"
        self.gender_menu = ctk.CTkOptionMenu(row_age_gender, values=gender_options, height=38, font=("Roboto", 13), corner_radius=8)
        self.gender_menu.set(default_gender)
        self.gender_menu.grid(row=1, column=1, sticky="ew", padx=(5, 0))

        # Notes / Session Tag
        ctk.CTkLabel(form, text="Session Notes / Tag (Optional)", font=("Roboto", 13), anchor="w").pack(fill="x", pady=(5, 2))
        self.entry_notes = ctk.CTkEntry(form, placeholder_text="e.g. Resting, Post-workout", height=38, font=("Roboto", 14), corner_radius=8)
        self.entry_notes.pack(fill="x", pady=(0, 15))
        if self.current_info.get("notes"):
            self.entry_notes.insert(0, self.current_info["notes"])

        self.lbl_error = ctk.CTkLabel(container, text="", font=("Roboto", 12), text_color="#e74c3c")
        self.lbl_error.pack(pady=(0, 10))

        # Buttons
        btn_frame = ctk.CTkFrame(container, fg_color="transparent")
        btn_frame.pack(fill="x", padx=15, pady=(5, 10))
        btn_frame.grid_columnconfigure(0, weight=1)
        btn_frame.grid_columnconfigure(1, weight=1)

        btn_cancel = ctk.CTkButton(btn_frame, text="Cancel", fg_color="#444444", hover_color="#555555",
                                   height=42, font=("Roboto", 14), corner_radius=10, command=self.cancel)
        btn_cancel.grid(row=0, column=0, sticky="ew", padx=(0, 5))

        btn_confirm = ctk.CTkButton(btn_frame, text="Confirm & Proceed", fg_color="#2ecc71", hover_color="#27ae60",
                                    height=42, font=("Roboto", 14, "bold"), corner_radius=10, command=self.confirm)
        btn_confirm.grid(row=0, column=1, sticky="ew", padx=(5, 0))

        # Focus name entry and bind Return key
        self.entry_name.focus_set()
        self.bind("<Return>", lambda e: self.confirm())
        self.bind("<Escape>", lambda e: self.cancel())

    def confirm(self):
        name = self.entry_name.get().strip()
        if not name:
            self.lbl_error.configure(text="Please enter a name or subject ID.")
            self.entry_name.focus_set()
            return

        gender = self.gender_menu.get()
        if gender == "Select Gender":
            gender = ""

        info = {
            "name": name,
            "age": self.entry_age.get().strip(),
            "gender": gender,
            "notes": self.entry_notes.get().strip()
        }
        self.submitted = True
        self.destroy()
        if self.on_submit:
            self.on_submit(info)

    def cancel(self):
        self.destroy()
