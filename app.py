import tkinter as tk
from win32api import GetSystemMetrics
import win32gui
import win32con
from ultralytics import YOLO
import pyautogui
import numpy as np
import cv2
from PIL import Image, ImageTk
from lib.manga_ocr import MangaOcr
from langdetect import detect
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import keyboard

class TextBoxLens(tk.Tk):
    def __init__(self, root):
        self.detect_model = YOLO(r"pretrained_models/best1.pt")  # pretrained YOLO12n model
        self.read_model = MangaOcr()
        # read_model = ""
        self.tokenizer_ja = AutoTokenizer.from_pretrained(r"D:\datatrain\ja_en")
        self.model_ja = AutoModelForSeq2SeqLM.from_pretrained(r"D:\datatrain\ja_en")
        self.tokenizer_zh = AutoTokenizer.from_pretrained(r"D:\datatrain\zh_en")
        self.model_zh = AutoModelForSeq2SeqLM.from_pretrained(r"D:\datatrain\zh_en")

        self.root = root

        self.bg_canvas = tk.Canvas(self.root, width=GetSystemMetrics(0), height=GetSystemMetrics(1), background='green', bd=0, highlightthickness=0)
        self.bg_canvas.pack()
        #Temp var
        self.screen_img = None
        self.list_temp_imgs = []
        self.temp_img = None

        self.hwnd = self.bg_canvas.winfo_id()
        self.setClickthrough()
        # self.get_bounding_boxes()

        #Key binds
        keyboard.add_hotkey('ctrl+space', self.get_bounding_boxes)
        keyboard.add_hotkey('ctrl+Esc', self.quit)
        keyboard.add_hotkey('ctrl+n', self.clear_screen)

    def setClickthrough(self):
        print("setting window properties")
        try:
            styles = win32gui.GetWindowLong(self.hwnd, win32con.GWL_EXSTYLE)
            styles = win32con.WS_EX_TRANSPARENT
            win32gui.SetWindowLong(self.hwnd, win32con.GWL_EXSTYLE, styles)
            # win32gui.SetLayeredWindowAttributes(hwnd, 0, 255, win32con.LWA_ALPHA)
        except Exception as e:
            print(e)

    def translate_text(self, text, tokenizer, model):
        input_ids = tokenizer.encode(text, return_tensors="pt")
        outputs = model.generate(input_ids)
        decoded = tokenizer.decode(outputs[0], skip_special_tokens=True)
        return decoded
    
    def get_wrapped_text(self, text: str,
                     line_length: int, fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.4, thickness=1):
        lines = ['']
        for word in text.split():
            line = f'{lines[-1]} {word}'.strip()
            if cv2.getTextSize(line, fontFace=fontFace, fontScale=fontScale, thickness=thickness)[0][0] <= line_length:
                lines[-1] = line
            else:
                lines.append(word)
        return lines
    
    def get_optimal_font_scale(self, text, width, font_face = cv2.FONT_HERSHEY_SIMPLEX, thickness = 1):
        for scale in reversed(range(0, 60, 1)):
            textSize = cv2.getTextSize(text, fontFace=font_face, fontScale=scale/10, thickness=thickness)
            new_width = textSize[0][0]
            line_height = textSize[0][1]
            if (new_width <= width+10 or scale/10 <= 0.3):
                return new_width, line_height, scale/10
        return 1

    def add_text_to_image(
        self,
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
    ):
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

    def replace_text(self, img):
        read_text = self.read_model(img)
        if not read_text:
            return np.asarray(img)
        try:
            detectLanguage = detect(read_text)
        except:
            return np.asarray(img)
        if detectLanguage == 'ja':
            translated_text = self.translate_text(read_text, self.tokenizer_ja, self.model_ja)
        elif 'zh' in detectLanguage:
            translated_text = self.translate_text(read_text, self.tokenizer_zh, self.model_zh)
        else: 
            return np.asarray(img)
        print(read_text, '-',detectLanguage,'->', translated_text)
        lines = self.get_wrapped_text(translated_text, img.size[0])
        lines = list(filter(None, lines))
        line_width, line_height, font_scale = self.get_optimal_font_scale(max(lines, key=len), img.size[0])
        max_w = line_width + 10 if line_width > img.size[0] else img.size[0]
        max_h = line_height*1.2*(len(lines)+1) if line_height*1.2*(len(lines)+1) > img.size[1] else img.size[1]
        empty_img = 255 * np.ones((int(max_h), int(max_w), 3), dtype=np.uint8)
        return self.add_text_to_image(
            empty_img,
            '\n'.join(lines),
            top_left_xy=(0, 0),
            font_scale=font_scale
        )
        # empty_img = Image.new("RGB", (new_w, new_h), (255,255,255))
        # new_img = ImageDraw.Draw(empty_img)
        # new_img.text((0, 0), '\n'.join(lines), fill="#000", font=font)
        # return empty_img
    
    def clear_screen(self):
        self.list_temp_imgs.clear()
        # clear the canvas 
        self.bg_canvas.delete('all') 
        self.bg_canvas.update()

    def get_bounding_boxes(self):
        # print("trigger")
        # clear_temp_dir()
        # global bbox_recs, abcd
        # print(self.list_temp_imgs)
        self.clear_screen()
        if not self.bg_canvas.winfo_children():
            self.screen_img = pyautogui.screenshot()
            # img.show()
            # Convert the screenshot to a numpy array
            frame = np.array(self.screen_img)

            # Convert it from BGR(Blue, Green, Red) to
            # RGB(Red, Green, Blue)
            detect_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            results = self.detect_model.predict(source=detect_frame, imgsz=1280, conf=0.1, classes = [0, 1], agnostic_nms=True) #
            bounding_boxes = results[0].boxes.xyxy.tolist()
            if bounding_boxes and len(bounding_boxes) != 0:
                bounding_boxes.sort(key=lambda x: x[1])
                for coor in bounding_boxes:
                    coor = list(map(int, coor)) #Convert float to int 
                    # temp_filename = 'temp/' + str(uuid.uuid4()) + '.png'
                    # cv2.imwrite(temp_filename, tempo_image)
                    # temp_image = ImageTk.PhotoImage(Image.open(temp_filename))
                    self.temp_image = ImageTk.PhotoImage(Image.fromarray(self.replace_text(self.screen_img.crop((coor[0],coor[1], coor[2],coor[3])))))
                    # tempo_image.show()
                    # print(read_model(img.crop((coor[0],coor[1], coor[2],coor[3]))))
                    self.list_temp_imgs.append(self.temp_image)
                    # abcd.append(tempo_image)
                    # # self.bg_canvas.create_rectangle(100, 100, 1000, 1000, fill='red')
                    self.bg_canvas.create_image(coor[0], coor[1], image = self.temp_image, anchor=tk.NW)
                    self.bg_canvas.update()
                self.bg_canvas.pack()
            # root.after(10000, get_bounding_boxes)  # reschedule event in 2 seconds

    def quit(self):
        self.root.destroy()
if __name__ == '__main__':
    # Dimensions
    width = GetSystemMetrics(0) #self.winfo_screenwidth()
    height = GetSystemMetrics(1) #self.winfo_screenheight()

    root = tk.Tk()
    root.geometry('%dx%d' % (width, height))
    root.attributes('-fullscreen', False)
    root.title("MangaLens")
    root.attributes("-topmost", 1)
    root.attributes('-transparentcolor', 'green')
    root.wm_attributes('-transparentcolor', 'green')
    root.wm_attributes("-topmost", 1)
    root.config(background='green') 
    # root.attributes("-alpha", 0.75)
    # root.overrideredirect(True)
    root.wm_attributes('-fullscreen', True)
    main_screen = TextBoxLens(root)
    




    # frame = ImageTk.PhotoImage(file="result.jpg")
    # bg.create_image(1920/2, 1080/2, image=frame)
    # bg.pack()
    root.mainloop()