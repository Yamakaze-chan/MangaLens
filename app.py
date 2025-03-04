import tkinter as tk
from win32api import GetSystemMetrics
from ultralytics import YOLO
import pyautogui
import numpy as np
import cv2
from PIL import Image, ImageTk, ImageDraw, ImageFont
import re

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
        if os.getenv('en_vi_weight'):
            self.tokenizer_en2vi = AutoTokenizer.from_pretrained(os.getenv('en_vi_token'))
            self.model_en2vi = AutoModelForSeq2SeqLM.from_pretrained(os.getenv('en_vi_weight'))
        else: 
            self.tokenizer_en2vi = None
            self.model_en2vi = None

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

        #Run OCR right after initialization complete
        # self.get_bounding_boxes()

        #Global key binding
        keyboard.add_hotkey('`', self.get_bounding_boxes)
        keyboard.add_hotkey('ctrl+alt', self.quit)
        keyboard.add_hotkey('Shift+`', self.clear_screen)

        keyboard.add_hotkey('Shift+W', self.getTranslateArea)

    def getTranslateArea(self):
        self.translate_area = measure_translate_area()

    def translate_text(self, 
                       text: str, 
                       tokenizer: AutoTokenizer, 
                       model: AutoModelForSeq2SeqLM) ->  str:
        input_ids = tokenizer.encode(text, return_tensors="pt")
        outputs = model.generate(input_ids)
        decoded = tokenizer.decode(outputs[0], skip_special_tokens=True)
        return decoded
    
    def translate_en2vi(self, en_text: str) -> str:
        input_ids = self.tokenizer_en2vi(en_text, return_tensors="pt").input_ids
        output_ids = self.model_en2vi.generate(
            input_ids,
            decoder_start_token_id = self.tokenizer_en2vi.lang_code_to_id["vi_VN"],
            num_return_sequences=1,
            # # With sampling
            # do_sample=True,
            # top_k=100,
            # top_p=0.8,
            # With beam search
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
            print(self.gettxtsize(text, font)[0], max_width)
            lines.append(text) 
        else:
            words = re.split(r'(\W+)', text)
            print(words)
            i = 0
            while i < len(words):
                line = ''         
                while i < len(words) and self.gettxtsize(line + words[i], font)[0] <= max_width - 5:   
                    line = line + words[i]
                    i += 1
                if not line:
                    line = words[i]
                    i += 1
                if line.strip() != '':
                    lines.append(line.strip())    
        return lines
    
    
    def draw_text(self, font, text, x, y, x_max, y_max):    
        image_size = x_max - x
        line_height = self.gettxtsize(text, font)[1] + 10
        print(x_max, y_max)
        lines = self.text_wrap(text,font, image_size)
        w = x_max - x
        h = y + (line_height*len(lines)) if y + (line_height*len(lines)) > y_max - y else y_max - y
        add_text_img = Image.fromarray(255 * np.ones((int(h), int(w), 3), dtype=np.uint8) )
        for line in lines:
            ImageDraw.Draw(add_text_img).text((x+5,y),line,font=font,fill=0)
            y = y + line_height
        return add_text_img

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
        if self.model_en2vi is not None and self.tokenizer_en2vi is not None:                           #| Uncomment these 2 lines to translate to Vietnamese
            translated_text = self.translate_en2vi(translated_text)                                     #|
        print(read_text, '-',detectLanguage,'->', translated_text)
        return self.draw_text(
            font = ImageFont.truetype(r"assets\arial.ttf",13),
            text = translated_text,
            x = 0,
            y = 0,
            x_max = img.size[0],
            y_max = img.size[1]
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