import tkinter as tk
import pyautogui
import numpy as np
import cv2
import time
import threading
from PIL import Image, ImageTk, ImageFont
from torchvision.ops import nms
import keyboard
from win32api import GetSystemMetrics

from utils import measure_translate_area


class TextBoxLens:
    def __init__(self, overlay_root, ui_panel, engine):
        self.root = overlay_root
        self.ui = ui_panel
        self.engine = engine
        self.auto_mode = False
        self.last_screenshot_hash = None
        self.is_processing = False 
        self.translate_area = (0, 0, GetSystemMetrics(0), GetSystemMetrics(1))
        
        self.bg_canvas = tk.Canvas(self.root, width=GetSystemMetrics(0), height=GetSystemMetrics(1), 
                                   background='green', bd=0, highlightthickness=0)
        self.bg_canvas.pack()

        self.screen_img = None
        self.list_temp_imgs = []
        self.temp_image = None

        # Đăng ký phím tắt hệ thống
        keyboard.add_hotkey('`', self.trigger_scan)
        keyboard.add_hotkey('ctrl+alt', self.quit_all)
        keyboard.add_hotkey('Shift+`', self.clear_screen)
        keyboard.add_hotkey('Shift+W', self.getTranslateArea)

    def trigger_scan(self):
        if self.is_processing: 
            return
        self.is_processing = True
        self.ui.update_status("Đang tự động dịch...", "orange")
        threading.Thread(target=self._run_logic_and_unlock, daemon=True).start()

    def _run_logic_and_unlock(self):
        try:
            self.get_bounding_boxes()
        finally:
            self.is_processing = False
            if self.auto_mode:
                self.ui.update_status("Auto: Đang chờ trang mới...", "cyan")
            else:
                self.ui.update_status("Sẵn sàng", "green")

    def clear_screen(self):
        self.list_temp_imgs.clear()
        self.bg_canvas.delete("all")
        self.bg_canvas.update()

    def getTranslateArea(self):
        self.ui.iconify()
        area = measure_translate_area()
        self.translate_area = area
        self.ui.deiconify()
        self.ui.update_area_label(area)

    def quit_all(self):
        self.root.quit()
        self.ui.destroy()

    def toggle_auto_mode(self, state):
        self.auto_mode = state
        if self.auto_mode:
            self.ui.update_status("Auto: Đang theo dõi...", "cyan")
            threading.Thread(target=self.auto_translate_loop, daemon=True).start()
        else:
            self.ui.update_status("Sẵn sàng", "green")

    def is_screen_changed(self):
        try:
            x, y, w, h = self.translate_area
            if w == 0 or h == 0: 
                return False
            
            current_img = pyautogui.screenshot(region=(x, y, w, h))
            current_img_cv = cv2.cvtColor(np.array(current_img), cv2.COLOR_RGB2GRAY)
            current_img_cv = cv2.resize(current_img_cv, (50, 50))

            if self.last_screenshot_hash is None:
                self.last_screenshot_hash = current_img_cv
                return True
            score = cv2.absdiff(self.last_screenshot_hash, current_img_cv).mean()
            self.last_screenshot_hash = current_img_cv
            return score > 5 
        except Exception:
            return False

    def auto_translate_loop(self):
        while self.auto_mode:
            if not self.is_processing:
                if self.is_screen_changed():
                    self.ui.after(0, self.trigger_scan)
            time.sleep(3)

    # Tìm hàm replace_text(self, img: Image) trong lens.py và chỉnh sửa:
    def replace_text(self, img: Image) -> Image:
        start_time = time.perf_counter()
        detectLanguage = self.engine.languages_detector.get_top_label(img)
        end_time = time.perf_counter()
        execution_time = (end_time - start_time) * 1000

        print(f"--- Ngôn ngữ phát hiện: {detectLanguage} ({execution_time:.2f} ms) ---")

        read_text = ""
        translated_text = ""
        
        if detectLanguage == 'Japanese':
            start_time = time.perf_counter()
            read_text = self.engine.jp_read_model(img)
            if not read_text:
                return img
            translated_text = self.engine.translate_ja2en(read_text)
            end_time = time.perf_counter()
            print(f"---JP OCR & Dịch: {(end_time - start_time)*1000:.2f} ms ---")
            
        elif detectLanguage == 'Chinese':
            start_time = time.perf_counter()
            read_text = self.engine.zh_read_model.get_ocr_text(img)
            if not read_text:
                return img
            translated_text = self.engine.translate_zh2en(read_text, self.engine.zh_en_model)
            end_time = time.perf_counter()
            print(f"---CN OCR & Dịch: {(end_time - start_time)*1000:.2f} ms ---")
            
        elif detectLanguage == 'Korean':
            start_time = time.perf_counter()
            read_text = self.engine.zh_read_model.get_ocr_text(img)
            if not read_text:
                return img
            translated_text = self.engine.translate_kr2en(read_text)
            end_time = time.perf_counter()
            print(f"---KR OCR & Dịch: {(end_time - start_time)*1000:.2f} ms ---")
        else:
            return img

        # Dịch từ tiếng Anh sang tiếng Việt nếu có nạp model dịch EN->VI
        if self.engine.model_en2vi is not None and self.engine.tokenizer_en2vi is not None:
            translated_text = self.engine.translate_en2vi(translated_text)
        
        if translated_text is None:
            translated_text = ""

        # ── GỬI DỮ LIỆU ĐỐI CHIẾU LÊN BẢNG ĐIỀU KHIỂN CHÍNH ───────────────────
        try:
            # Truy cập thông qua đối tượng giao diện để đưa vào bảng lịch sử log
            self.ui.after(0, lambda: self.ui.add_history_log(read_text, translated_text))
        except Exception as e:
            print(f"Lỗi gửi dữ liệu lịch sử dịch: {e}")
        # ─────────────────────────────────────────────────────────────────────

        # Thay vì arial.ttf gây lỗi dấu tiếng Việt, sử dụng font segoeui đã tối ưu
        import os
        system_font_path = r"C:\Windows\Fonts\segoeui.ttf"
        if not os.path.exists(system_font_path):
            system_font_path = "arial.ttf"

        return self.engine.draw_text(
            font=ImageFont.truetype(system_font_path, 13),
            text=translated_text,
            x=0,
            y=0,
            x_max=img.size[0],
            y_max=img.size[1]
        )
    
    def get_bounding_boxes(self) -> None:
        self.clear_screen()
        if not self.bg_canvas.winfo_children():
            self.screen_img = pyautogui.screenshot(region=self.translate_area)
            frame = np.array(self.screen_img)
            detect_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.engine.detect_model.predict(
                source=detect_frame, imgsz=1280, conf=0.2, classes=[0, 1], agnostic_nms=True, iou=0.5
            )
            boxes = results[0].boxes.xyxy
            scores = results[0].boxes.conf

            keep_indices = nms(boxes, scores, iou_threshold=0.3)
            bounding_boxes = boxes[keep_indices].tolist()

            if bounding_boxes and len(bounding_boxes) != 0:
                bounding_boxes.sort(key=lambda x: x[1])
                for coor in bounding_boxes:
                    coor = list(map(int, coor))
                    cropped = self.screen_img.crop((coor[0], coor[1], coor[2], coor[3]))
                    replaced_image = self.replace_text(cropped)
                    
                    self.temp_image = ImageTk.PhotoImage(replaced_image)
                    self.list_temp_imgs.append(self.temp_image)
                    self.bg_canvas.create_image(
                        coor[0] + self.translate_area[0], 
                        coor[1] + self.translate_area[1], 
                        image=self.temp_image, 
                        anchor=tk.NW
                    )
                    self.bg_canvas.update()
                self.bg_canvas.pack()