import os
os.environ['FLAGS_use_mkldnn'] = '0'
import tkinter as tk
import customtkinter as ctk
from win32api import GetSystemMetrics
from ultralytics import YOLO
import pyautogui
import numpy as np
import cv2
from PIL import Image, ImageTk, ImageDraw, ImageFont
import re
from transformers import pipeline, T5ForConditionalGeneration, T5Tokenizer
import torch
from torchvision.ops import nms
import threading
import time
import re
from lib.quickmt_zh_en.translator import Translator
# import pycorrector as zh_corrector
# from pycorrector import MacBertCorrector
from lib.mangatranslator.translator import MangaTranslator
from lib.manga_predictor.MangaLanguageDetector import MangaLanguageDetector
from llama_cpp import Llama
from lib.kr_ocr.kr_ocr import KoreanOCR
from lib.kr_en.KR_Translator import KoreanTranslator
import unicodedata

try:
    from lib.manga_ocr import MangaOcr #You can install through PIP. Please read this Github repo for more information - https://github.com/kha-white/manga-ocr
except:
    from manga_ocr import MangaOcr
try:
    from lib.lingua import Language, LanguageDetectorBuilder
except:
    from lingua import Language, LanguageDetectorBuilder
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, AutoModelForCausalLM
import keyboard
from dotenv import load_dotenv
load_dotenv()

# Thiết lập giao diện
ctk.set_appearance_mode("dark")
ctk.set_default_color_theme("blue")

APP_ICON_PATH = "assets/app-logo.ico"          # Window icon (.ico)
SPLASH_LOGO_PATH = "assets/splash-art.jpg"       # Splash logo image (.png / .jpg)

def measure_translate_area():
    root = tk.Tk()
    root.wm_attributes("-topmost", 1)
    root.config(background='gray') 
    root.wm_attributes('-alpha', 0.5)
    root.wm_attributes('-fullscreen', True)
    root.attributes('-transparentcolor', 'green')    
    
    start_x = [None]
    start_y = [None]
    rect_id = [None]
    result = [None]
    
    canvas = tk.Canvas(root, width=GetSystemMetrics(0), height=GetSystemMetrics(1), bg='black')
    canvas.pack()
    
    def on_press(event):
        start_x[0] = event.x
        start_y[0] = event.y
        rect_id[0] = canvas.create_rectangle(start_x[0], start_y[0], 
                                           start_x[0], start_y[0],
                                           outline='blue', dash=(5, 5), fill="green")
    
    def on_drag(event):
        if start_x[0] is not None and rect_id[0] is not None:
            canvas.coords(rect_id[0], start_x[0], start_y[0], event.x, event.y)
    
    def on_release(event):
        if start_x[0] is not None:
            x = min(start_x[0], event.x)
            y = min(start_y[0], event.y)
            width = abs(event.x - start_x[0])
            height = abs(event.y - start_y[0])
            result[0] = (x, y, width, height)
            canvas.delete(rect_id[0])
            root.quit()
    
    canvas.bind('<Button-1>', on_press)
    canvas.bind('<B1-Motion>', on_drag)
    canvas.bind('<ButtonRelease-1>', on_release)
    
    root.mainloop()
    root.destroy()
    return result[0] if result[0] is not None else (0, 0, 0, 0)

class TextBoxLens(tk.Tk):
    def __init__(self, overlay_root, ui_panel):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.ui = ui_panel
        self.auto_mode = False
        self.last_screenshot_hash = None
        self.is_processing = False 
        
        # Load model weights
        self.detect_model = YOLO(os.getenv('YOLO_weight'))
        self.languages_detector = MangaLanguageDetector(os.getenv('MANGA_LANGUAGE_DETECTION_MODEL'), device='cpu')
        self.jp_read_model = MangaOcr(os.getenv('MangaOCR_weight'))
        self.zh_read_model = MangaTranslator(model_path=os.getenv('ZH_READ_MODEL'))
        self.ja_model_path = os.getenv('JA_TRANS_MODEL')
        self.ja_llm = Llama(
            model_path=self.ja_model_path,
            n_ctx=2048,
            n_threads=1,
            verbose=False,
            n_batch=8,
            stop=["\n"],
            echo=False
        )

        self.zh_en_model = Translator(
            os.getenv('ZH_TRANS_MODEL'), 
            device=self.device
        )
        self.kr_read_model = KoreanOCR()
        self.kr_en_model = KoreanTranslator(model_name=os.getenv('KR_TRANS_MODEL'))
        self.languages = [Language.JAPANESE, Language.CHINESE]
        self.detector = LanguageDetectorBuilder.from_languages(*self.languages).build()
        
        if os.getenv('en_vi_weight'):
            self.tokenizer_en2vi = AutoTokenizer.from_pretrained(os.getenv('en_vi_token'))
            self.model_en2vi = AutoModelForSeq2SeqLM.from_pretrained(os.getenv('en_vi_weight'))
        else: 
            self.tokenizer_en2vi = None
            self.model_en2vi = None

        self.root = overlay_root
        self.translate_area = (0, 0, GetSystemMetrics(0), GetSystemMetrics(1))
        
        self.bg_canvas = tk.Canvas(self.root, width=GetSystemMetrics(0), height=GetSystemMetrics(1), 
                                   background='green', bd=0, highlightthickness=0)
        self.bg_canvas.pack()

        self.screen_img = None
        self.list_temp_imgs = []
        self.temp_img = None

        # Bind hotkeys
        keyboard.add_hotkey('`', self.trigger_scan)
        keyboard.add_hotkey('ctrl+alt', self.quit_all)
        keyboard.add_hotkey('Shift+`', self.clear_screen)
        keyboard.add_hotkey('Shift+W', self.getTranslateArea)

    def trigger_scan(self):
        self.ui.update_status("Đang quét...", "orange")
        threading.Thread(target=self.get_bounding_boxes, daemon=True).start()

    def clear_screen(self):
        self.bg_canvas.delete("all")

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
            if w == 0 or h == 0: return False
            
            current_img = pyautogui.screenshot(region=(x, y, w, h))
            current_img_cv = cv2.cvtColor(np.array(current_img), cv2.COLOR_RGB2GRAY)
            current_img_cv = cv2.resize(current_img_cv, (50, 50))

            if self.last_screenshot_hash is None:
                self.last_screenshot_hash = current_img_cv
                return True
            score = cv2.absdiff(self.last_screenshot_hash, current_img_cv).mean()
            self.last_screenshot_hash = current_img_cv
            return score > 5 
        except:
            return False

    def auto_translate_loop(self):
        """Vòng lặp chạy ngầm kiểm tra màn hình"""
        while self.auto_mode:
            if not self.is_processing:
                if self.is_screen_changed():
                    self.ui.after(0, self.trigger_scan)
            time.sleep(3)

    def trigger_scan(self):
        if self.is_processing: return
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

    def translate_text(self, 
                       text: str, 
                       tokenizer: AutoTokenizer, 
                       model: AutoModelForSeq2SeqLM) ->  str:
        input_ids = tokenizer.encode(text, return_tensors="pt")
        outputs = model.generate(input_ids)
        decoded = tokenizer.decode(outputs[0], skip_special_tokens=True)
        return decoded

    def translate_ja2en(self, 
                        text: str,   
                        target_lang: str = "en") -> str:
        """
        Dịch văn bản sử dụng LiquidAI model.
        :param text: Văn bản cần dịch
        :param target_lang: 'en' để dịch sang tiếng Anh, 'ja' để dịch sang tiếng Nhật
        """
        if not text.strip():
            return text
        # 1. Xác định System Prompt dựa trên ngôn ngữ đích
        if target_lang.lower() == "en":
            # Thêm các chỉ dẫn cụ thể để fix lỗi Katakana và từ cảm thán
            system_prompt = "Translate to English"
        elif target_lang.lower() == "ja":
            system_prompt = "Translate to Japanese."
        else:
            system_prompt = "Translate accurately."
        clean_text = re.sub(r'[\r\n]+', ' ', text)
        clean_text = unicodedata.normalize('NFKC', clean_text).strip()
        self.ja_llm.reset()
        response = self.ja_llm.create_chat_completion(
        messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": clean_text}
            ],
            temperature=0.1,
            max_tokens=128,
            repeat_penalty=1.1
        )

        translated_text = response['choices'][0]['message']['content'].strip()
        print("original text:", clean_text, " -> translated text:", translated_text)

        return translated_text
    
    # def translate_zh2en(
    #     self,
    #     text: str,
    #     tokenizer: T5Tokenizer,
    #     model: T5ForConditionalGeneration,
    #     target_lang: str = "en",
    # ) -> str:
    #     """Dịch văn bản sử dụng mô hình T5 (Seq2Seq).

    #     :param text: Văn bản cần dịch
    #     :param target_lang: 'en' để dịch sang tiếng Anh, 'zh' để dịch sang tiếng Trung
    #     """
    #     if not text.strip():
    #         return text

    #     # 1. Xác định Prefix dựa trên ngôn ngữ đích (Đặc thù của dòng T5)
    #     if target_lang.lower() == "en":
    #         prefix = "translate to en: "
    #     elif target_lang.lower() == "zh":
    #         prefix = "translate to zh: "
    #     else:
    #         # Dự phòng trường hợp bạn muốn dịch sang tiếng Nga 'ru'
    #         prefix = f"translate to {target_lang.lower()}: "

    #     src_text = prefix + text

    #     # 2. Tokenize và đẩy lên device (GPU/CPU)
    #     # T5Tokenizer trả về dict chuẩn nên không cần check Quirks như LiquidAI
    #     inputs = tokenizer(src_text, return_tensors="pt").to(self.device)

    #     # 3. Generate
    #     # Đối với T5 (Dịch thuật), greedy search (do_sample=False) hoặc beam search thường ra kết quả chính xác hơn.
    #     # Nhưng nếu bạn thích sáng tạo chút thì giữ do_sample=True.
    #     with torch.no_grad():
    #         output_ids = model.generate(
    #             **inputs,
    #             max_new_tokens=128,
    #             temperature=0.3,  # Thường để thấp cho dịch thuật chính xác
    #             do_sample=False,  # Đổi thành True nếu bạn muốn dịch đa dạng hơn
    #             repetition_penalty=1.05,
    #         )

    #     # 4. Decode
    #     # LƯU Ý: Với T5, output_ids[0] chính là văn bản được dịch luôn,
    #     # không chứa lại prompt đầu vào nên không cần cắt [input_length:]
    #     translated_text = tokenizer.decode(
    #         output_ids[0], skip_special_tokens=True
    #     ).strip()

    #     # 5. Clean up (giữ lại logic từ hàm cũ của bạn nếu có)
    #     if hasattr(self, "clean_translation"):
    #         translated_text = self.clean_translation(translated_text)

    #     print("original text:", text, " -> translated text:", translated_text)

    #     return translated_text
    
    def translate_zh2en(self, 
                        text: str, 
                        translator: any,
                        beam_size: int = 5) -> str:
        """
        Dịch văn bản Trung -> Anh sử dụng QuickMT.
        :param text: Văn bản tiếng Trung cần dịch
        :param translator: Đối tượng Translator đã được khởi tạo
        :param beam_size: 5 cho chất lượng tốt, 1 cho tốc độ nhanh
        """
        if not text.strip():
            return text

        try:
            translated_text = translator(text, beam_size=beam_size,
                patience=1,
                length_penalty=0.8,
                coverage_penalty=1.0,
                repetition_penalty=1.2)

            if isinstance(translated_text, list):
                translated_text = translated_text[0]
            
            translated_text = str(translated_text).strip()
                
            print(f"original text: {text} -> translated text: {translated_text}")

            return translated_text

        except Exception as e:
            print(f"Lỗi khi dịch QuickMT: {e}")
            return text
        
    def translate_kr2en(
        self,
        text: str,
        beam_size: int = 5
    ) -> str:
        if not text or not text.strip():
            return text

        try:
            translated_text = self.kr_en_model.translate(
                text=text
            )

            translated_text = str(translated_text).strip()

            print(f"original text: {text} -> translated text: {translated_text}")

            return translated_text

        except Exception as e:
            print(f"Lỗi khi dịch KR->EN: {e}")
            return text

    def translate_en2vi(self, en_text: str) -> str:
        input_ids = self.tokenizer_en2vi(en_text, return_tensors="pt").input_ids
        output_ids = self.model_en2vi.generate(
            input_ids,
            decoder_start_token_id = self.tokenizer_en2vi.lang_code_to_id["vi_VN"],
            num_return_sequences=1,
            num_beams=5,
            early_stopping=True
        )
        vi_text = self.tokenizer_en2vi.batch_decode(output_ids, skip_special_tokens=True)
        vi_text = " ".join(vi_text)
        return vi_text
    
    def get_wrapped_text(self, 
                        text: str,
                        line_length: int, 
                        fontFace: int = cv2.FONT_HERSHEY_SIMPLEX, 
                        fontScale: float = 0.4, 
                        thickness: int = 1) -> list:
        lines = ['']
        for word in text.split():
            line = f'{lines[-1]} {word}'.strip()
            if cv2.getTextSize(line, fontFace=fontFace, fontScale=fontScale, thickness=thickness)[0][0] <= line_length:
                lines[-1] = line
            else:
                lines.append(word)
        return lines

    def gettxtsize(self, text, font):
        left, top, right, bottom = font.getbbox(text)
        width = right - left
        height = bottom - top
        return width, height

    def text_wrap(self, text, font, max_width):
        lines = []
        if self.gettxtsize(text, font)[0] <= max_width - 5:
            lines.append(text)
        else:
            try:
                words = re.split(r'(\W+)', text)
                # Remove empty strings from split result
                words = [w for w in words if w]
                i = 0
                while i < len(words):
                    line = ''
                    while i < len(words) and self.gettxtsize(line + words[i], font)[0] <= max_width - 5:
                        line = line + words[i]
                        i += 1

                    while i < len(words) and re.match(r'^\W+$', words[i]) and not words[i].strip() == '':
                        line = line + words[i]
                        i += 1

                    if not line:
                        line = words[i]
                        i += 1
                    if line.strip() != '':
                        lines.append(line.strip())
            except Exception as e:
                print(f"An error occurred: {e}")
            finally:
                return lines
    
    
    def draw_text(self, font, text, x, y, x_max, y_max):
        if not text:
            return Image.new('RGB', (x_max - x, y_max - y), (255, 255, 255))

        frame_width = round((x_max - x) * 1.1)
        frame_height = round((y_max - y) * 1.1)

        padding_x = int(frame_width * 0.1)
        padding_y = int(frame_height * 0.1)

        usable_width = frame_width - (2 * padding_x)
        usable_height = frame_height - (2 * padding_y)

        current_font = font
        MIN_FONT_SIZE = 10

        while True:
            lines = self.text_wrap(text, current_font, usable_width)
            if not lines:
                break

            line_height = int(current_font.getbbox("Ay")[3])
            total_text_height = len(lines) * line_height

            if total_text_height <= usable_height:
                break

            current_size = None
            try:
                current_size = current_font.size
            except AttributeError:
                pass
            if current_size is None:
                try:
                    current_size = current_font.font.size
                except AttributeError:
                    break

            if current_size <= MIN_FONT_SIZE:
                break

            new_size = max(current_size - 1, MIN_FONT_SIZE)
            try:
                current_font = ImageFont.truetype(current_font.path, new_size)
            except (AttributeError, OSError):
                break

        lines = self.text_wrap(text, current_font, usable_width)
        if not lines:
            return Image.new('RGB', (1, 1), (255, 255, 255))

        max_line_width = max(current_font.getbbox(line)[2] for line in lines)

        if max_line_width > usable_width:
            usable_width = max_line_width
            padding_x = int(usable_width * 0.1)
            frame_width = usable_width + (2 * padding_x)

        line_height = int(current_font.getbbox("Ay")[3])
        total_text_height = len(lines) * line_height

        if total_text_height > usable_height:
            frame_height = total_text_height + (2 * padding_y)

        # --- Draw ---
        add_text_img = Image.new('RGB', (frame_width, frame_height), (255, 255, 255))
        draw = ImageDraw.Draw(add_text_img)

        # border_color = (0, 0, 0)
        # border_width = 5
        # draw.rectangle([(0, 0), (frame_width - 1, frame_height - 1)],
        #             outline=border_color, width=border_width)

        start_y = max(padding_y, (frame_height - total_text_height) // 2)
        current_y = start_y

        for line in lines:
            draw.text((padding_x, current_y), line, font=current_font, fill=(0, 0, 0))
            current_y += line_height

        return add_text_img
    
    def replace_text(self, img: Image) -> np.array:
        start_time = time.perf_counter()
        detectLanguage = self.languages_detector.get_top_label(img)
        end_time = time.perf_counter()
        execution_time = (end_time - start_time) * 1000

        print(f"--- Language Detector Execution Time: {execution_time:.2f} ms; Language Detected: {detectLanguage} ---")

        if detectLanguage == 'Japanese':
            start_time = time.perf_counter()
            read_text = self.jp_read_model(img)
            if not read_text:
                return img
            
            if read_text is None:
                read_text = ""
            translated_text = self.translate_ja2en(read_text)
            end_time = time.perf_counter()
            execution_time = (end_time - start_time) * 1000
            print(f"---JP Language Execution Time: {execution_time:.2f} ms ---")
        elif  detectLanguage == 'Chinese':
            start_time = time.perf_counter()
            read_text = self.zh_read_model.get_ocr_text(img)
            if not read_text:
                return img
            
            if read_text is None:
                read_text = ""
            translated_text = self.translate_zh2en(read_text, self.zh_en_model)
            end_time = time.perf_counter()
            execution_time = (end_time - start_time) * 1000
            print(f"---CN Language Execution Time: {execution_time:.2f} ms ---")
        elif  detectLanguage == 'Korean':
            start_time = time.perf_counter()
            read_text = self.zh_read_model.get_ocr_text(img)
            print("kr:",read_text)
            if not read_text:
                return img
            
            if read_text is None:
                read_text = ""
            translated_text = self.translate_kr2en(read_text, self.kr_en_model)
            end_time = time.perf_counter()
            execution_time = (end_time - start_time) * 1000
            print(f"---KR Language Execution Time: {execution_time:.2f} ms ---")
        else:
            return img

        if self.model_en2vi is not None and self.tokenizer_en2vi is not None:
            translated_text = self.translate_en2vi(translated_text)
        
        if translated_text is None:
            translated_text = ""

        return self.draw_text(
            font = ImageFont.truetype(r"assets\arial.ttf", 13),
            text = translated_text,
            x = 0,
            y = 0,
            x_max = img.size[0],
            y_max = img.size[1]
        )

    def clear_screen(self) -> None:
        self.list_temp_imgs.clear()
        self.bg_canvas.delete('all') 
        self.bg_canvas.update()

    def get_bounding_boxes(self) -> None:
        self.clear_screen()
        if not self.bg_canvas.winfo_children():
            self.screen_img = pyautogui.screenshot(region=self.translate_area)
            frame = np.array(self.screen_img)
            detect_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.detect_model.predict(source=detect_frame, imgsz=1280, conf=0.2, classes = [0, 1], agnostic_nms=True, iou=0.5)
            boxes = results[0].boxes.xyxy
            scores = results[0].boxes.conf

            keep_indices = nms(boxes, scores, iou_threshold=0.3)

            bounding_boxes = boxes[keep_indices].tolist()
            if bounding_boxes and len(bounding_boxes) != 0:
                bounding_boxes.sort(key=lambda x: x[1])
                for coor in bounding_boxes:
                    coor = list(map(int, coor))
                    self.temp_image = ImageTk.PhotoImage(self.replace_text(self.screen_img.crop((coor[0],coor[1], coor[2],coor[3]))))
                    self.list_temp_imgs.append(self.temp_image)
                    self.bg_canvas.create_image(coor[0] + self.translate_area[0], 
                                                coor[1] + self.translate_area[1], 
                                                image = self.temp_image, 
                                                anchor=tk.NW)
                    self.bg_canvas.update()
                self.bg_canvas.pack()
            # root.after(2000, get_bounding_boxes)  # reschedule event in 2 seconds

    def quit(self) -> None:
        self.root.destroy()

# Lớp Giao diện điều khiển mới bằng CustomTkinter
class SplashScreen(ctk.CTkToplevel):
    """Animated loading screen shown while the model initialises."""
 
    def __init__(self, parent):
        super().__init__(parent)
        self.title("")
        self.geometry("320x340")
        self.resizable(False, False)
        self.overrideredirect(True)          # borderless
        self.attributes("-topmost", True)
        self._center()
 
        # ── background ──────────────────────────────────────────────────────
        self.configure(fg_color="#0d0d0d")
 
        # ── logo / icon area ─────────────────────────────────────────────────
        if os.path.exists(SPLASH_LOGO_PATH):
            img = Image.open(SPLASH_LOGO_PATH).resize((96, 96), Image.LANCZOS)
            self._logo_img = ctk.CTkImage(light_image=img, dark_image=img, size=(96, 96))
            ctk.CTkLabel(self, image=self._logo_img, text="").pack(pady=(36, 8))
        else:
            # Placeholder glyph when no logo file is found
            ctk.CTkLabel(
                self, text="🔍", font=ctk.CTkFont(size=56)
            ).pack(pady=(36, 8))
 
        # ── app name ─────────────────────────────────────────────────────────
        ctk.CTkLabel(
            self,
            text="MangaLens",
            font=ctk.CTkFont(family="Segoe UI", size=26, weight="bold"),
            text_color="#ffffff",
        ).pack()
 
        ctk.CTkLabel(
            self,
            text="AI-powered manga translation",
            font=ctk.CTkFont(family="Segoe UI", size=11),
            text_color="#666666",
        ).pack(pady=(2, 20))
 
        # ── progress bar ─────────────────────────────────────────────────────
        self._progress = ctk.CTkProgressBar(
            self, width=240, height=6,
            corner_radius=3,
            fg_color="#1e1e1e",
            progress_color="#4fc3f7",
        )
        self._progress.pack(pady=(0, 8))
        self._progress.set(0)
 
        # ── status text ──────────────────────────────────────────────────────
        self._status_var = tk.StringVar(value="Initialising…")
        ctk.CTkLabel(
            self,
            textvariable=self._status_var,
            font=ctk.CTkFont(family="Segoe UI", size=10),
            text_color="#555555",
        ).pack()
 
        self._animate_progress()
 
    # ── helpers ──────────────────────────────────────────────────────────────
 
    def _center(self):
        sw = GetSystemMetrics(0)
        sh = GetSystemMetrics(1)
        self.geometry(f"320x340+{(sw - 320)//2}+{(sh - 340)//2}")
 
    def _animate_progress(self, value: float = 0.0):
        """Smoothly fill the bar to ~85 % while loading; caller sets to 1.0."""
        if value < 0.85:
            self._progress.set(value)
            self.after(35, self._animate_progress, value + 0.012)
 
    def finish(self, callback=None):
        """Fill bar to 100 %, pause, then destroy and run callback."""
        self._progress.set(1.0)
        self._status_var.set("Ready!")
        self.after(600, lambda: self._teardown(callback))
 
    def _teardown(self, callback):
        self.destroy()
        if callback:
            callback()
 
    def set_status(self, text: str):
        self._status_var.set(text)
 
 
class ControlPanel(ctk.CTk):
    def __init__(self):
        super().__init__()
 
        # ── window setup ─────────────────────────────────────────────────────
        self.title("MangaLens Control")
        self.geometry("320x450")
        self.attributes("-topmost", True)
 
        # ── custom icon ───────────────────────────────────────────────────────
        #   Replace APP_ICON_PATH at the top of this file with your .ico path.
        if os.path.exists(APP_ICON_PATH):
            self.iconbitmap(APP_ICON_PATH)
 
        # Hide main window until splash is done
        self.withdraw()
 
        # ── overlay window (translation layer) ───────────────────────────────
        self.overlay = tk.Toplevel()
        self.overlay.withdraw()             # also hidden during load
        self.setup_overlay_window(self.overlay)
 
        # ── show splash & start background init ──────────────────────────────
        self.splash = SplashScreen(self)
        threading.Thread(target=self._init_logic_threaded, daemon=True).start()
 
        # ── build main UI (off-screen while hidden) ───────────────────────────
        self._build_ui()
 
    # ── UI construction ───────────────────────────────────────────────────────
 
    def _build_ui(self):
        self.label = ctk.CTkLabel(
            self, text="MANGA LENS UI",
            font=ctk.CTkFont(size=20, weight="bold"),
        )
        self.label.pack(pady=20)
 
        self.status_label = ctk.CTkLabel(
            self, text="Status: Loading model…", text_color="yellow"
        )
        self.status_label.pack(pady=5)
 
        self.area_info = ctk.CTkLabel(self, text="Area: Fullscreen", font=("Arial", 11))
        self.area_info.pack(pady=5)
 
        self.btn_area = ctk.CTkButton(
            self, text="Select Translation Area (Shift+W)",
            command=lambda: self.lens.getTranslateArea(),
        )
        self.btn_area.pack(pady=10, padx=20, fill="x")
 
        self.btn_scan = ctk.CTkButton(
            self, text="Translate Now ( ` )",
            fg_color="green", hover_color="darkgreen",
            command=lambda: self.lens.trigger_scan(),
        )
        self.btn_scan.pack(pady=10, padx=20, fill="x")
 
        self.btn_clear = ctk.CTkButton(
            self, text="Clear Screen (Shift+`)",
            fg_color="gray",
            command=lambda: self.lens.clear_screen(),
        )
        self.btn_clear.pack(pady=10, padx=20, fill="x")
 
        self.btn_quit = ctk.CTkButton(
            self, text="Quit (Ctrl+Alt)",
            fg_color="red", hover_color="darkred",
            command=lambda: self.lens.quit_all(),
        )
        self.btn_quit.pack(pady=20, padx=20, fill="x")
 
        self.auto_frame = ctk.CTkFrame(self, fg_color="transparent")
        self.auto_frame.pack(pady=10, padx=20, fill="x")
 
        self.auto_switch = ctk.CTkSwitch(
            self.auto_frame, text="AUTO TRANSLATE PAGE",
            command=self.handle_auto_switch,
            font=ctk.CTkFont(weight="bold"),
        )
        self.auto_switch.pack(side="left")
 
    # ── overlay setup ─────────────────────────────────────────────────────────
 
    def setup_overlay_window(self, win):
        width = GetSystemMetrics(0)
        height = GetSystemMetrics(1)
        win.geometry("%dx%d+0+0" % (width, height))
        win.overrideredirect(True)
        win.attributes("-topmost", True)
        win.attributes("-transparentcolor", "green")
        win.config(background="green")
 
    # ── background initialisation ─────────────────────────────────────────────
 
    def _init_logic_threaded(self):
        """Runs in a worker thread; posts UI updates back to main thread."""
        self.after(0, lambda: self.splash.set_status("Loading OCR model…"))
        time.sleep(1.0)                     # ← replace with real model load
 
        self.after(0, lambda: self.splash.set_status("Loading translation engine…"))
        time.sleep(1.0)                     # ← replace with real model load
 
        self.after(0, lambda: self.splash.set_status("Starting overlay…"))
        self.lens = TextBoxLens(self.overlay, self)   # real init here
 
        # Hand off to main thread to finish splash and reveal the app
        self.after(0, self._on_init_complete)
 
    def _on_init_complete(self):
        self.splash.finish(callback=self._show_main_window)
 
    def _show_main_window(self):
        self.overlay.deiconify()
        self.deiconify()
        self.update_status("Sẵn sàng", "green")
 
    # ── public helpers ────────────────────────────────────────────────────────
 
    def update_status(self, text, color):
        self.status_label.configure(text=f"Trạng thái: {text}", text_color=color)
 
    def update_area_label(self, area):
        self.area_info.configure(text=f"Vùng: {area[2]}x{area[3]} tại {area[0]},{area[1]}")
 
    def handle_auto_switch(self):
        is_on = self.auto_switch.get() == 1
        self.lens.toggle_auto_mode(is_on)
 
 
if __name__ == "__main__":
    app = ControlPanel()
    app.mainloop()