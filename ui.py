import os
import tkinter as tk
import customtkinter as ctk
from PIL import Image
import threading
import time
from win32api import GetSystemMetrics
from config import APP_VERSION, UPDATE_JSON_URL
import webbrowser
import urllib.request
import json

from config import APP_ICON_PATH, SPLASH_LOGO_PATH
from engine import TranslationEngine
from lens import TextBoxLens

ctk.set_appearance_mode("dark")
ctk.set_default_color_theme("blue")

# Từ điển hỗ trợ đa ngôn ngữ
LANG_TEXTS = {
    "VI": {
        "sub_logo": "Trình dịch Desktop",
        "btn_scan": " Dịch Ngay ( ` )",
        "btn_area": " Chọn vùng (Shift+W)",
        "btn_clear": " Xóa đè (Shift+`)",
        "btn_quit": " Đóng ứng dụng",
        "status_title": "TRẠNG THÁI",
        "auto_title": "TỰ ĐỘNG DỊCH",
        "auto_switch": "Theo dõi",
        "area_fullscreen": "Vùng: Toàn màn hình",
        "area_prefix": "Phân vùng quét",
        "area_coord": "Góc tọa độ",
        "history_title": "BẢNG ĐỐI CHIẾU CÂU DỊCH",
        "history_orig": "GỐC",
        "history_trans": "DỊCH",
        "status_ready": "Sẵn sàng",
        "status_init": "Đang khởi tạo...",
    },
    "EN": {
        "sub_logo": "Desktop Translator",
        "btn_scan": " Translate Now ( ` )",
        "btn_area": " Select Area (Shift+W)",
        "btn_clear": " Clear Overlay (Shift+`)",
        "btn_quit": " Quit Application",
        "status_title": "STATUS",
        "auto_title": "AUTO TRANSLATE",
        "auto_switch": "Monitor",
        "area_fullscreen": "Area: Fullscreen",
        "area_prefix": "Scan area",
        "area_coord": "Coordinates",
        "history_title": "TRANSLATION COMPARISON",
        "history_orig": "ORIGINAL",
        "history_trans": "TRANSLATION",
        "status_ready": "Ready",
        "status_init": "Initializing...",
    }
}


class FloatingWidget(ctk.CTkToplevel):
    """Thanh công cụ nổi nhỏ gọn ở góc màn hình khi thu nhỏ ứng dụng chính."""
    def __init__(self, parent):
        super().__init__(parent)
        self.parent = parent
        self.overrideredirect(True)
        self.attributes("-topmost", True)
        self.configure(fg_color="#181825")
        
        # Mở rộng kích thước từ 190 lên 225 để chứa thêm nút chọn vùng
        self.geometry("225x40+10+10")

        # Logic kéo thả (Drag and Drop) widget nổi trên màn hình
        self.bind("<Button-1>", self._start_drag)
        self.bind("<B1-Motion>", self._on_drag)

        # Chấm tròn hiển thị trạng thái hoạt động (Xanh/Cam)
        self.status_dot = ctk.CTkLabel(
            self, text="●", font=ctk.CTkFont(size=14), text_color="#a6e3a1"
        )
        self.status_dot.pack(side="left", padx=(10, 5))

        # Khởi tạo nhãn trạng thái dựa trên ngôn ngữ hiện tại của cha
        initial_status = LANG_TEXTS[self.parent.current_lang]["status_ready"]
        self.status_text = ctk.CTkLabel(
            self, text=initial_status, font=ctk.CTkFont(family="Segoe UI", size=11, weight="bold"),
            text_color="#cdd6f4"
        )
        self.status_text.pack(side="left", padx=2)

        # Nút khôi phục lại giao diện lớn
        self.btn_restore = ctk.CTkButton(
            self,
            image=ctk.CTkImage(
                light_image=Image.open(r"assets\open_icon.png"),
                dark_image=Image.open(r"assets\open_icon.png"),
                size=(18, 18)
            ),
            text="", width=26, height=26, corner_radius=13,
            fg_color="#313244", hover_color="#45475a",
            font=ctk.CTkFont(size=12, weight="bold"),
            command=self.parent.restore_dashboard
        )
        self.btn_restore.pack(side="right", padx=(2, 5))

        # Nút chọn vùng quét trực tiếp trên widget nổi
        self.btn_area = ctk.CTkButton(
            self,
            image=ctk.CTkImage(
                light_image=Image.open(r"assets\zone.png"),
                dark_image=Image.open(r"assets\zone.png"),
                size=(16, 16)
            ),
            text="",
            width=26,
            height=26,
            corner_radius=13,
            fg_color="#313244",
            hover_color="#45475a",
            command=self.parent.start_area_selection
        )
        self.btn_area.pack(side="right", padx=2)

        # Nút dịch nhanh trực tiếp trên widget nổi
        self.btn_quick_scan = ctk.CTkButton(
            self,
            image=ctk.CTkImage(
                light_image=Image.open(r"assets\translate_icon.png"),
                dark_image=Image.open(r"assets\translate_icon.png"),
                size=(18, 18)
            ),
            text="",
            width=26,
            height=26,
            corner_radius=13,
            fg_color="#313244",
            hover_color="#45475a",
            command=lambda: self.parent.lens.trigger_scan()
        )
        self.btn_quick_scan.pack(side="right", padx=2)

    def _start_drag(self, event):
        self._drag_start_x = event.x
        self._drag_start_y = event.y

    def _on_drag(self, event):
        x = self.winfo_x() - self._drag_start_x + event.x
        y = self.winfo_y() - self._drag_start_y + event.y
        self.geometry(f"+{x}+{y}")

    def update_status(self, text, color_hex):
        self.status_text.configure(text=text)
        self.status_dot.configure(text_color=color_hex)


class SplashScreen(ctk.CTkToplevel):
    def __init__(self, parent):
        super().__init__(parent)
        self.title("")
        self.geometry("340x360")
        self.resizable(False, False)
        self.overrideredirect(True)
        self.attributes("-topmost", True)
        self._center()
        self.configure(fg_color="#11111b")
 
        if os.path.exists(SPLASH_LOGO_PATH):
            img = Image.open(SPLASH_LOGO_PATH).resize((110, 110), Image.LANCZOS)
            self._logo_img = ctk.CTkImage(light_image=img, dark_image=img, size=(110, 110))
            ctk.CTkLabel(self, image=self._logo_img, text="").pack(pady=(40, 10))
        else:
            ctk.CTkLabel(self, text="🔮", font=ctk.CTkFont(size=64)).pack(pady=(40, 10))
 
        ctk.CTkLabel(
            self, text="MangaLens Pro",
            font=ctk.CTkFont(family="Segoe UI", size=28, weight="bold"),
            text_color="#cdd6f4",
        ).pack()
 
        ctk.CTkLabel(
            self, text="Hệ thống dịch thuật Manga bằng AI",
            font=ctk.CTkFont(family="Segoe UI", size=12),
            text_color="#a6adc8",
        ).pack(pady=(2, 20))
 
        self._progress = ctk.CTkProgressBar(
            self, width=260, height=5,
            corner_radius=10,
            fg_color="#1e1e2e",
            progress_color="#89b4fa",
        )
        self._progress.pack(pady=(0, 8))
        self._progress.set(0)
 
        self._status_var = tk.StringVar(value="Đang khởi tạo hệ thống…")
        ctk.CTkLabel(
            self, textvariable=self._status_var,
            font=ctk.CTkFont(family="Segoe UI", size=10),
            text_color="#7f849c",
        ).pack()
 
        self._animate_progress()
 
    def _center(self):
        sw = GetSystemMetrics(0)
        sh = GetSystemMetrics(1)
        self.geometry(f"340x360+{(sw - 340)//2}+{(sh - 360)//2}")
 
    def _animate_progress(self, value: float = 0.0):
        if value < 0.85:
            self._progress.set(value)
            self.after(35, self._animate_progress, value + 0.015)
 
    def finish(self, callback=None):
        self._progress.set(1.0)
        self._status_var.set("Hệ thống đã sẵn sàng!")
        self.after(500, lambda: self._teardown(callback))
 
    def _teardown(self, callback):
        self.destroy()
        if callback:
            callback()
 
    def set_status(self, text: str):
        self._status_var.set(text)

class UpdateDialog(ctk.CTkToplevel):
    """Cửa sổ thông báo khi phát hiện phiên bản mới."""
    def __init__(self, parent, latest_version, changelog, download_url):
        super().__init__(parent)
        self.title("Bản cập nhật mới")
        self.geometry("400x320")
        self.resizable(False, False)
        self.attributes("-topmost", True)
        self.configure(fg_color="#11111b")
        self._center()

        # Tiêu đề thông báo
        self.title_lbl = ctk.CTkLabel(
            self, text="Đã có phiên bản mới!",
            font=ctk.CTkFont(family="Segoe UI", size=18, weight="bold"),
            text_color="#a6e3a1"
        )
        self.title_lbl.pack(pady=(20, 5))

        self.ver_lbl = ctk.CTkLabel(
            self, text=f"Phiên bản hiện tại: {APP_VERSION}  ➔  Mới nhất: {latest_version}",
            font=ctk.CTkFont(family="Segoe UI", size=11),
            text_color="#a6adc8"
        )
        self.ver_lbl.pack(pady=2)

        # Khung hiển thị nội dung cập nhật (Changelog)
        self.changelog_frame = ctk.CTkFrame(self, fg_color="#181825", corner_radius=6)
        self.changelog_frame.pack(fill="both", expand=True, padx=20, pady=10)

        self.cl_title = ctk.CTkLabel(
            self.changelog_frame, text="Nội dung thay đổi:",
            font=ctk.CTkFont(family="Segoe UI", size=11, weight="bold"),
            text_color="#89b4fa"
        )
        self.cl_title.pack(padx=12, pady=(8, 2), anchor="w")

        self.cl_box = ctk.CTkTextbox(
            self.changelog_frame, fg_color="transparent", text_color="#cdd6f4",
            font=ctk.CTkFont(family="Segoe UI", size=11), wrap="word"
        )
        self.cl_box.pack(fill="both", expand=True, padx=8, pady=(0, 8))
        self.cl_box.insert("0.0", changelog)
        self.cl_box.configure(state="disabled")

        # Hàng nút bấm hành động
        self.btn_row = ctk.CTkFrame(self, fg_color="transparent")
        self.btn_row.pack(fill="x", side="bottom", pady=15, padx=20)

        # Nút "Để sau"
        self.btn_skip = ctk.CTkButton(
            self.btn_row, text="Để sau", width=100, height=32,
            fg_color="#313244", hover_color="#45475a", text_color="#cdd6f4",
            command=self.destroy
        )
        self.btn_skip.pack(side="left")

        # Nút "Cập nhật ngay"
        self.btn_update = ctk.CTkButton(
            self.btn_row, text="Cập nhật ngay", width=140, height=32,
            fg_color="#a6e3a1", hover_color="#94e2d5", text_color="#11111b",
            font=ctk.CTkFont(family="Segoe UI", size=12, weight="bold"),
            command=lambda: self._open_download_page(download_url)
        )
        self.btn_update.pack(side="right")

    def _center(self):
        sw = GetSystemMetrics(0)
        sh = GetSystemMetrics(1)
        self.geometry(f"400x320+{(sw - 400)//2}+{(sh - 320)//2}")

    def _open_download_page(self, url):
        try:
            webbrowser.open(url)
            self.destroy()
        except Exception as e:
            print(f"Không thể mở liên kết: {e}")

class ControlPanel(ctk.CTk):
    def __init__(self):
        super().__init__()
 
        self.title("MangaLens Dashboard")
        self.geometry("640x520")  # Mở rộng không gian hiển thị
        self.resizable(False, False)
        self.attributes("-topmost", True)
        self.configure(fg_color="#11111b")
 
        if os.path.exists(APP_ICON_PATH):
            self.iconbitmap(APP_ICON_PATH)
 
        self.withdraw()
 
        self.overlay = tk.Toplevel()
        self.overlay.withdraw()
        self.setup_overlay_window(self.overlay)
 
        # Cấu hình ngôn ngữ mặc định và thuộc tính lưu trữ phân vùng
        self.current_lang = "VI"
        self.current_area = None
        self._was_minimized_before_select = False  # Cờ theo dõi trạng thái thu nhỏ

        # Khởi tạo Widget nổi (ẩn ban đầu)
        self.floating_widget = None

        # Bắt sự kiện thu nhỏ để kích hoạt Widget nổi
        self.bind("<Unmap>", self._on_minimize)
 
        self.splash = SplashScreen(self)
        threading.Thread(target=self._init_logic_threaded, daemon=True).start()
 
        self._build_ui()

    def _init_logic_threaded(self):
        self.after(0, lambda: self.splash.set_status("Đang nạp công cụ dịch thuật…"))
        self.engine = TranslationEngine()
 
        self.after(0, lambda: self.splash.set_status("Khởi tạo lớp phủ đồ họa…"))
        self.lens = TextBoxLens(self.overlay, self, self.engine)
 
        # Bắt đầu luồng kiểm tra cập nhật phiên bản
        threading.Thread(target=self._check_update_threaded, daemon=True).start()
 
        self.after(0, self._on_init_complete)

    def _check_update_threaded(self):
        """Hàm chạy ngầm gửi yêu cầu kiểm tra phiên bản từ máy chủ."""
        try:
            req = urllib.request.Request(
                UPDATE_JSON_URL, 
                headers={'User-Agent': 'MangaLensUpdater/1.0'}
            )
            with urllib.request.urlopen(req, timeout=8) as response:
                data = json.loads(response.read().decode("utf-8"))
                latest_version = data.get("latest_version")
                download_url = data.get("download_url")
                changelog = data.get("changelog", "Không có thông tin thay đổi.")

                if self._is_version_newer(APP_VERSION, latest_version):
                    self.after(0, lambda: self._show_update_dialog(latest_version, changelog, download_url))
        except Exception as e:
            print(f"Kiểm tra cập nhật thất bại: {e}")

    def _is_version_newer(self, current: str, latest: str) -> bool:
        """Hàm so sánh dạng Semantic Versioning (X.Y.Z)."""
        try:
            curr_parts = [int(x) for x in current.split(".")]
            late_parts = [int(x) for x in latest.split(".")]
            return late_parts > curr_parts
        except Exception:
            return False

    def _show_update_dialog(self, latest_version, changelog, download_url):
        """Khởi tạo cửa sổ thông báo cập nhật."""
        dialog = UpdateDialog(self, latest_version, changelog, download_url)
        dialog.focus()
 
    def _build_ui(self):
        self.grid_columnconfigure(0, weight=1, minsize=190)
        self.grid_columnconfigure(1, weight=2)
        self.grid_rowconfigure(0, weight=1)

        # ── 1. SIDEBAR PANEL (Bên trái) ──────────────────────────────────────
        self.sidebar_frame = ctk.CTkFrame(self, fg_color="#181825", corner_radius=0)
        self.sidebar_frame.grid(row=0, column=0, sticky="nsew", padx=0, pady=0)
        
        self.logo_label = ctk.CTkLabel(
            self.sidebar_frame, text="MangaLens", 
            font=ctk.CTkFont(family="Segoe UI", size=20, weight="bold"),
            text_color="#89b4fa"
        )
        self.logo_label.pack(padx=20, pady=(25, 2), anchor="w")
        
        # Nhãn dịch vụ phụ trợ
        self.sub_logo = ctk.CTkLabel(
            self.sidebar_frame, text=LANG_TEXTS[self.current_lang]["sub_logo"], 
            font=ctk.CTkFont(family="Segoe UI", size=10),
            text_color="#585b70"
        )
        self.sub_logo.pack(padx=20, pady=(0, 10), anchor="w")

        # Nút chuyển đổi ngôn ngữ Việt - Anh
        self.lang_switch = ctk.CTkSegmentedButton(
            self.sidebar_frame,
            values=["VI", "EN"],
            command=self._change_language,
            font=ctk.CTkFont(family="Segoe UI", size=11, weight="bold"),
            fg_color="#1e1e2e",
            selected_color="#89b4fa",
            unselected_color="#313244",
        )
        self.lang_switch.pack(padx=15, pady=(0, 15), fill="x")
        self.lang_switch.set(self.current_lang)

        self.btn_scan = ctk.CTkButton(
            self.sidebar_frame,
            image=ctk.CTkImage(
                light_image=Image.open(r"assets\translate_icon_d.png"),
                dark_image=Image.open(r"assets\translate_icon_d.png"),
                size=(20, 20)
            ),
            compound="left",
            text=LANG_TEXTS[self.current_lang]["btn_scan"],
            font=ctk.CTkFont(family="Segoe UI", size=12, weight="bold"),
            fg_color="#a6e3a1",
            text_color="#11111b",
            hover_color="#94e2d5",
            height=40,
            command=lambda: self.lens.trigger_scan(),
        )
        self.btn_scan.pack(pady=10, padx=15, fill="x")

        self.btn_area = ctk.CTkButton(
            self.sidebar_frame,
            image=ctk.CTkImage(
                light_image=Image.open(r"assets\zone.png"),
                dark_image=Image.open(r"assets\zone.png"),
                size=(18, 18)
            ),
            compound="left",
            text=LANG_TEXTS[self.current_lang]["btn_area"],
            font=ctk.CTkFont(family="Segoe UI", size=12),
            fg_color="#313244",
            text_color="#cdd6f4",
            hover_color="#45475a",
            height=32,
            command=self.start_area_selection,
        )
        self.btn_area.pack(pady=5, padx=15, fill="x")

        self.btn_clear = ctk.CTkButton(
            self.sidebar_frame,
            image=ctk.CTkImage(
                light_image=Image.open(r"assets\clear.png"),
                dark_image=Image.open(r"assets\clear.png"),
                size=(18, 18)
            ),
            compound="left",
            text=LANG_TEXTS[self.current_lang]["btn_clear"],
            font=ctk.CTkFont(family="Segoe UI", size=12),
            fg_color="#313244",
            text_color="#cdd6f4",
            hover_color="#45475a",
            height=32,
            command=lambda: self.lens.clear_screen(),
        )
        self.btn_clear.pack(pady=5, padx=15, fill="x")

        self.btn_quit = ctk.CTkButton(
            self.sidebar_frame,
            image=ctk.CTkImage(
                light_image=Image.open(r"assets\close.png"),
                dark_image=Image.open(r"assets\close.png"),
                size=(16, 16)
            ),
            compound="left",
            text=LANG_TEXTS[self.current_lang]["btn_quit"],
            font=ctk.CTkFont(family="Segoe UI", size=11),
            fg_color="#f38ba8",
            text_color="#11111b",
            hover_color="#eba0ac",
            height=28,
            command=lambda: self.lens.quit_all(),
        )
        self.btn_quit.pack(side="bottom", pady=20, padx=15, fill="x")


        # ── 2. MAIN PANEL (Bên phải) ───────────────────────────────────────
        self.main_frame = ctk.CTkFrame(self, fg_color="transparent")
        self.main_frame.grid(row=0, column=1, sticky="nsew", padx=20, pady=20)

        # Thẻ trạng thái & Tùy chọn nhanh
        self.top_row = ctk.CTkFrame(self.main_frame, fg_color="transparent")
        self.top_row.pack(fill="x", pady=(0, 10))

        # Status Box
        self.status_card = ctk.CTkFrame(self.top_row, fg_color="#1e1e2e", corner_radius=8, border_width=1, border_color="#313244")
        self.status_card.pack(side="left", fill="both", expand=True, padx=(0, 5))
        
        self.card1_title = ctk.CTkLabel(
            self.status_card, text=LANG_TEXTS[self.current_lang]["status_title"], 
            font=ctk.CTkFont(family="Segoe UI", size=10, weight="bold"), text_color="#7f849c"
        )
        self.card1_title.pack(padx=12, pady=(8, 1), anchor="w")
        
        self.status_label = ctk.CTkLabel(
            self.status_card, text=LANG_TEXTS[self.current_lang]["status_init"], 
            font=ctk.CTkFont(family="Segoe UI", size=14, weight="bold"), text_color="#f9e2af"
        )
        self.status_label.pack(padx=12, pady=(0, 8), anchor="w")

        # Auto Switch Box
        self.auto_card = ctk.CTkFrame(self.top_row, fg_color="#1e1e2e", corner_radius=8, border_width=1, border_color="#313244")
        self.auto_card.pack(side="right", fill="both", expand=True, padx=(5, 0))
        
        self.card3_title = ctk.CTkLabel(
            self.auto_card, text=LANG_TEXTS[self.current_lang]["auto_title"], 
            font=ctk.CTkFont(family="Segoe UI", size=10, weight="bold"), text_color="#7f849c"
        )
        self.card3_title.pack(padx=12, pady=(8, 3), anchor="w")

        self.auto_switch = ctk.CTkSwitch(
            self.auto_card, text=LANG_TEXTS[self.current_lang]["auto_switch"],
            command=self.handle_auto_switch,
            font=ctk.CTkFont(family="Segoe UI", size=11),
            progress_color="#89b4fa"
        )
        self.auto_switch.pack(padx=12, pady=(0, 8), anchor="w")

        # Thẻ thông tin vùng quét
        self.area_card = ctk.CTkFrame(self.main_frame, fg_color="#1e1e2e", corner_radius=8, border_width=1, border_color="#313244")
        self.area_card.pack(fill="x", pady=5)
        
        self.area_info = ctk.CTkLabel(
            self.area_card, text=LANG_TEXTS[self.current_lang]["area_fullscreen"], 
            font=ctk.CTkFont(family="Consolas", size=11), text_color="#a6adc8"
        )
        self.area_info.pack(padx=12, pady=6, anchor="w")

        # ── KHU VỰC BẢNG ĐỐI CHIẾU CÂU DỊCH (Translation History) ────────────
        self.history_card = ctk.CTkFrame(self.main_frame, fg_color="#1e1e2e", corner_radius=8, border_width=1, border_color="#313244")
        self.history_card.pack(fill="both", expand=True, pady=(10, 5))

        self.history_title = ctk.CTkLabel(
            self.history_card, text=LANG_TEXTS[self.current_lang]["history_title"], 
            font=ctk.CTkFont(family="Segoe UI", size=11, weight="bold"), text_color="#89b4fa"
        )
        self.history_title.pack(padx=15, pady=(10, 5), anchor="w")

        # Bảng Textbox hiển thị log
        self.history_box = ctk.CTkTextbox(
            self.history_card, 
            fg_color="#11111b",
            text_color="#cdd6f4",
            font=ctk.CTkFont(family="Segoe UI", size=12),
            wrap="word",
            corner_radius=6
        )
        self.history_box.pack(fill="both", expand=True, padx=15, pady=(0, 15))
        self.history_box.configure(state="disabled")

    def start_area_selection(self):
        """Hàm trung gian chuẩn bị trước khi kích hoạt chọn vùng."""
        # Ghi nhận xem Dashboard chính có đang thu nhỏ (không hiển thị) hay không
        self._was_minimized_before_select = not self.winfo_viewable()
        
        # Ẩn tạm thời thanh công cụ nổi để tránh cản trở tầm nhìn khi chọn vùng
        if self.floating_widget:
            self.floating_widget.withdraw()
            
        self.lens.getTranslateArea()

    def _change_language(self, lang_code):
        """Hàm thay đổi ngôn ngữ giao diện chính và đồng bộ hóa các nhãn."""
        self.current_lang = lang_code
        texts = LANG_TEXTS[lang_code]

        self.sub_logo.configure(text=texts["sub_logo"])
        self.btn_scan.configure(text=texts["btn_scan"])
        self.btn_area.configure(text=texts["btn_area"])
        self.btn_clear.configure(text=texts["btn_clear"])
        self.btn_quit.configure(text=texts["btn_quit"])

        self.card1_title.configure(text=texts["status_title"])
        self.card3_title.configure(text=texts["auto_title"])
        self.auto_switch.configure(text=texts["auto_switch"])
        self.history_title.configure(text=texts["history_title"])

        if self.current_area is not None:
            self.update_area_label(self.current_area)
        else:
            self.area_info.configure(text=texts["area_fullscreen"])

        current_status_text = self.status_label.cget("text")
        if lang_code == "EN":
            if current_status_text in ["Sẵn sàng", "Ready"]:
                self.update_status(texts["status_ready"], "#a6e3a1")
            elif current_status_text in ["Đang khởi tạo...", "Initializing..."]:
                self.update_status(texts["status_init"], "#f9e2af")
        else:
            if current_status_text in ["Ready", "Sẵn sàng"]:
                self.update_status(texts["status_ready"], "#a6e3a1")
            elif current_status_text in ["Initializing...", "Đang khởi tạo..."]:
                self.update_status(texts["status_init"], "#f9e2af")

    def add_history_log(self, original: str, translated: str):
        """Hàm cập nhật câu gốc và câu dịch lên bảng đối chiếu."""
        if not original.strip():
            return
        
        self.history_box.configure(state="normal")
        label_orig = LANG_TEXTS[self.current_lang]["history_orig"]
        label_trans = LANG_TEXTS[self.current_lang]["history_trans"]

        self.history_box.insert("end", f"{label_orig}: {original.strip()}\n")
        self.history_box.insert("end", f"{label_trans}: {translated.strip()}\n")
        self.history_box.insert("end", "──────────────────────\n")
        self.history_box.see("end")
        self.history_box.configure(state="disabled")

    def setup_overlay_window(self, win):
        width = GetSystemMetrics(0)
        height = GetSystemMetrics(1)
        win.geometry("%dx%d+0+0" % (width, height))
        win.overrideredirect(True)
        win.attributes("-topmost", True)
        win.attributes("-transparentcolor", "green")
        win.config(background="green")
 
    def _on_init_complete(self):
        self.splash.finish(callback=self._show_main_window)
 
    def _show_main_window(self):
        self.overlay.deiconify()
        self.deiconify()
        self.update_status(LANG_TEXTS[self.current_lang]["status_ready"], "#a6e3a1")
 
    def _on_minimize(self, event):
        """Sự kiện khi người dùng nhấn nút thu nhỏ dashboard."""
        if event.widget == self:
            self.withdraw()  # Ẩn dashboard hoàn toàn khỏi Taskbar
            
            # Khởi tạo và hiển thị thanh widget nổi
            if self.floating_widget is None:
                self.floating_widget = FloatingWidget(self)
            else:
                self.floating_widget.deiconify()

    def restore_dashboard(self):
        """Khôi phục lại giao diện điều khiển chính."""
        if self.floating_widget:
            self.floating_widget.withdraw()
        self.deiconify()
        self.state("normal")
        self.overlay.deiconify()

    def update_status(self, text, color_hex):
        display_text = text
        if text in ["Sẵn sàng", "Ready"]:
            display_text = LANG_TEXTS[self.current_lang]["status_ready"]
        elif text in ["Đang khởi tạo...", "Initializing..."]:
            display_text = LANG_TEXTS[self.current_lang]["status_init"]

        self.status_label.configure(text=display_text, text_color=color_hex)
        if self.floating_widget:
            self.floating_widget.update_status(display_text, color_hex)
 
    def update_area_label(self, area):
        self.current_area = area
        prefix = LANG_TEXTS[self.current_lang]["area_prefix"]
        coord_label = LANG_TEXTS[self.current_lang]["area_coord"]
        self.area_info.configure(
            text=f"{prefix}: {area[2]}x{area[3]} ({coord_label}: {area[0]},{area[1]})"
        )
        
        # Nếu trước đó chọn vùng từ trạng thái thu nhỏ, trả lại giao diện thanh nổi gọn nhẹ
        if self._was_minimized_before_select:
            self.after(100, self._restore_minimized_state)

    def _restore_minimized_state(self):
        """Chuyển trạng thái giao diện về dạng thu nhỏ (FloatingWidget)."""
        self.withdraw()
        if self.floating_widget:
            self.floating_widget.deiconify()
        self._was_minimized_before_select = False
 
    def handle_auto_switch(self):
        is_on = self.auto_switch.get() == 1
        self.lens.toggle_auto_mode(is_on)