import tkinter as tk
from win32api import GetSystemMetrics
import win32gui
import win32con
from ultralytics import YOLO
import pyautogui
import numpy as np
import cv2
from PIL import Image, ImageTk

try:
    from lib.manga_ocr import MangaOcr #You can install through PIP. Please read this Github repo for more information - https://github.com/kha-white/manga-ocr
except:
    from manga_ocr import MangaOcr
try:
    from lib.lingua import Language, LanguageDetectorBuilder
except:
    from lingua import Language, LanguageDetectorBuilder
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import keyboard
import os
from dotenv import load_dotenv
load_dotenv()

def measure_translate_area():
    """
    Measures the size of a drag operation and returns (x, y, w, h).
    
    Returns:
        tuple: (x, y, width, height) of the dragged rectangle
    """
    root = tk.Tk()
    root.wm_attributes("-topmost", 1)
    root.config(background='gray') 
    root.wm_attributes('-alpha', 0.5)
    root.wm_attributes('-fullscreen', True)
    root.attributes('-transparentcolor', 'green')    # Color green will become transparent
    root.wm_attributes('-transparentcolor', 'green')
    
    # Variables to store coordinates and rectangle
    start_x = [None]  # Using lists to allow modification in nested functions
    start_y = [None]
    rect_id = [None]
    result = [None]   # To store the return value
    
    # Create a canvas to draw on
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
            # Calculate coordinates and size
            x = min(start_x[0], event.x)  # Top-left x
            y = min(start_y[0], event.y)  # Top-left y
            width = abs(event.x - start_x[0])
            height = abs(event.y - start_y[0])
            
            # Store result
            result[0] = (x, y, width, height)
            
            # Clean up
            canvas.delete(rect_id[0])
            start_x[0] = None
            start_y[0] = None
            rect_id[0] = None
            
            # Exit the mainloop
            root.quit()
    
    # Bind mouse events
    canvas.bind('<Button-1>', on_press)
    canvas.bind('<B1-Motion>', on_drag)
    canvas.bind('<ButtonRelease-1>', on_release)
    
    # Run the event loop
    root.mainloop()
    
    # Clean up and return
    root.destroy()
    return result[0] if result[0] is not None else (0, 0, 0, 0)

class TextBoxLens(tk.Tk):
    def __init__(self, root):
        #Load model weights
        self.detect_model = YOLO(os.getenv('YOLO_weight'))
        self.read_model = MangaOcr(os.getenv('MangaOCR_weight'))
        self.tokenizer_ja = AutoTokenizer.from_pretrained(os.getenv('ja_en_token'))
        self.model_ja = AutoModelForSeq2SeqLM.from_pretrained(os.getenv('ja_en_weight'))
        self.tokenizer_zh = AutoTokenizer.from_pretrained(os.getenv('zh_en_token'))
        self.model_zh = AutoModelForSeq2SeqLM.from_pretrained(os.getenv('zh_en_weight'))
        self.languages = [Language.JAPANESE, Language.CHINESE]
        self.detector = LanguageDetectorBuilder.from_languages(*self.languages).build()

        #Set root
        self.root = root

        #Translate area
        self.translate_area = (0, 0, GetSystemMetrics(0), GetSystemMetrics(1))
        
        #Create Canvas to add translations
        self.bg_canvas = tk.Canvas(self.root, width=GetSystemMetrics(0), height=GetSystemMetrics(1), background='green', bd=0, highlightthickness=0)
        self.bg_canvas.pack()

        #Temp var to prevent Python garbage collection
        self.screen_img = None
        self.list_temp_imgs = []
        self.temp_img = None

        #Make background and Canvas can be click through
        # self.hwnd = self.bg_canvas.winfo_id()
        # self.setClickthrough()

        #Run OCR right after initialization complete
        # self.get_bounding_boxes()

        #Global key binding
        keyboard.add_hotkey('`', self.get_bounding_boxes)
        keyboard.add_hotkey('ctrl+alt', self.quit)
        keyboard.add_hotkey('Shift+`', self.clear_screen)

        # keyboard.add_hotkey('Shift+Q', self.setClickthrough)
        keyboard.add_hotkey('Shift+W', self.getTranslateArea)

    # def setClickthrough(self):
    #     print("setting window properties")
    #     try:
    #         styles = win32gui.GetWindowLong(self.hwnd, win32con.GWL_EXSTYLE)
    #         styles = win32con.WS_EX_TRANSPARENT
    #         win32gui.SetWindowLong(self.hwnd, win32con.GWL_EXSTYLE, styles)
    #         # win32gui.SetLayeredWindowAttributes(hwnd, 0, 255, win32con.LWA_ALPHA)
    #     except Exception as e:
    #         print(e)

    def getTranslateArea(self):
        # print(measure_drag_size())
        self.translate_area = measure_translate_area()

    def translate_text(self, 
                       text: str, 
                       tokenizer: AutoTokenizer, 
                       model: AutoModelForSeq2SeqLM) ->  str:
        input_ids = tokenizer.encode(text, return_tensors="pt")
        outputs = model.generate(input_ids)
        decoded = tokenizer.decode(outputs[0], skip_special_tokens=True)
        return decoded
    
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
    
    def get_optimal_font_scale(self, 
                               text: str, 
                               width: int, 
                               font_face: int = cv2.FONT_HERSHEY_SIMPLEX, 
                               thickness: int = 1) -> int | int | float:
        for scale in reversed(range(0, 60, 1)):
            textSize = cv2.getTextSize(text, fontFace=font_face, fontScale=scale/10, thickness=thickness)
            new_width = textSize[0][0]
            line_height = textSize[0][1]
            if (new_width <= width+10 or scale/10 <= 0.3):
                return new_width, line_height, scale/10
        return width, 0, 1

    def add_text_to_image(self,
        image_rgb: np.ndarray,
        label: str,
        top_left_xy: tuple = (0, 0),
        font_scale: float = 1,
        font_thickness: float = 1,
        font_face=cv2.FONT_HERSHEY_SIMPLEX,
        font_color_rgb: tuple = (0, 0, 0),
        bg_color_rgb: tuple | None = None,
        outline_color_rgb: tuple | None = None,
        line_spacing: float = 1,
    ) -> np.array:
        """
        Adds text (including multi line text) to images.
        You can also control background color, outline color, and line spacing.

        outline color and line spacing adopted from: https://gist.github.com/EricCousineau-TRI/596f04c83da9b82d0389d3ea1d782592
        """
        OUTLINE_FONT_THICKNESS = 3 * font_thickness

        im_h, im_w = image_rgb.shape[:2]

        for line in label.splitlines():
            x, y = top_left_xy

            # ====== get text size
            if outline_color_rgb is None:
                get_text_size_font_thickness = font_thickness
            else:
                get_text_size_font_thickness = OUTLINE_FONT_THICKNESS

            (line_width, line_height_no_baseline), baseline = cv2.getTextSize(
                line,
                font_face,
                font_scale,
                get_text_size_font_thickness,
            )
            line_height = line_height_no_baseline + baseline

            if bg_color_rgb is not None and line:
                # === get actual mask sizes with regard to image crop
                if im_h - (y + line_height) <= 0:
                    sz_h = max(im_h - y, 0)
                else:
                    sz_h = line_height

                if im_w - (x + line_width) <= 0:
                    sz_w = max(im_w - x, 0)
                else:
                    sz_w = line_width

                # ==== add mask to image
                if sz_h > 0 and sz_w > 0:
                    bg_mask = np.zeros((sz_h, sz_w, 3), np.uint8)
                    bg_mask[:, :] = np.array(bg_color_rgb)
                    image_rgb[
                        y : y + sz_h,
                        x : x + sz_w,
                    ] = bg_mask

            # === add outline text to image
            if outline_color_rgb is not None:
                image_rgb = cv2.putText(
                    image_rgb,
                    line,
                    (x, y + line_height_no_baseline),  # putText start bottom-left
                    font_face,
                    font_scale,
                    outline_color_rgb,
                    OUTLINE_FONT_THICKNESS,
                    cv2.LINE_AA,
                )
            # === add text to image
            image_rgb = cv2.putText(
                image_rgb,
                line,
                (x, y + line_height_no_baseline),  # putText start bottom-left
                font_face,
                font_scale,
                font_color_rgb,
                font_thickness,
                cv2.LINE_AA,
            )
            top_left_xy = (x, y + int(line_height * line_spacing))

        return image_rgb

    def replace_text(self, img: Image) -> np.array:
        read_text = self.read_model(img)
        if not read_text: # Not detect text
            return np.asarray(img)
        try:
            detectLanguage = self.detector.detect_language_of((read_text)).iso_code_639_1.name.lower()
            # print(read_text, '-->', detectLanguage)
        except Exception as e: # Not detect language
            print(e)
            return np.asarray(img)
        if detectLanguage == 'ja':
            translated_text = self.translate_text(read_text, self.tokenizer_ja, self.model_ja)
        elif 'zh' in detectLanguage:
            translated_text = self.translate_text(read_text, self.tokenizer_zh, self.model_zh)
        else: # Language not defined
            return np.asarray(img)
        print(read_text, '-',detectLanguage,'->', translated_text)
        lines = self.get_wrapped_text(translated_text, img.size[0])
        lines = list(filter(None, lines))                                                                       #________
        line_width, line_height, font_scale = self.get_optimal_font_scale(max(lines, key=len), img.size[0])     #        \
        max_w = line_width + 10 if line_width > img.size[0] else img.size[0]                                    #         \ Trying to make the largest place to put text
        max_h = line_height*1.2*(len(lines)+1) if line_height*1.2*(len(lines)+1) > img.size[1] else img.size[1] #         /
        empty_img = 255 * np.ones((int(max_h), int(max_w), 3), dtype=np.uint8)                                  #________/
        return self.add_text_to_image(
            empty_img,
            '\n'.join(lines),
            top_left_xy=(0, 0),
            font_scale=font_scale
        )
    
    def clear_screen(self) -> None:
        #Clear the temp list
        self.list_temp_imgs.clear()
        # clear the canvas 
        self.bg_canvas.delete('all') 
        self.bg_canvas.update()

    def get_bounding_boxes(self) -> None:
        self.clear_screen()
        if not self.bg_canvas.winfo_children():
            self.screen_img = pyautogui.screenshot(region=self.translate_area)
            # Convert the screenshot to a numpy array
            frame = np.array(self.screen_img)
            # Convert it from BGR(Blue, Green, Red) to
            # RGB(Red, Green, Blue)
            detect_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.detect_model.predict(source=detect_frame, imgsz=1280, conf=0.2, classes = [0, 1], agnostic_nms=True, iou=0.5) #Get predicted bounding boxes
            bounding_boxes = results[0].boxes.xyxy.tolist() # Get coor
            if bounding_boxes and len(bounding_boxes) != 0:
                bounding_boxes.sort(key=lambda x: x[1]) # Sort based on top of bboxes
                for coor in bounding_boxes:
                    coor = list(map(int, coor)) #Convert float to int 
                    self.temp_image = ImageTk.PhotoImage(Image.fromarray(self.replace_text(self.screen_img.crop((coor[0],coor[1], coor[2],coor[3])))))
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

if __name__ == '__main__':
    # Get screen size
    width = GetSystemMetrics(0)
    height = GetSystemMetrics(1)

    root = tk.Tk()
    #App config
    root.geometry('%dx%d' % (width, height))
    root.attributes('-fullscreen', False)
    root.title("MangaLens")
    root.attributes("-topmost", 1)
    root.attributes('-transparentcolor', 'green')    # Color green will become transparent
    root.wm_attributes('-transparentcolor', 'green')
    root.wm_attributes("-topmost", 1)
    root.config(background='green') 
    root.wm_attributes('-fullscreen', True)

    main_screen = TextBoxLens(root)

    root.mainloop()