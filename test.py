# import tkinter as tk

# # Create a transparent fullscreen window
# root = tk.Tk()
# root.attributes('-fullscreen', False)  # Fullscreen mode
# root.attributes('-topmost', True)     # Stay on top of other windows
# root.attributes('-alpha', 0.5)       # Near-transparent (adjust as needed)
# root.configure(bg='gray')             # Background (mostly invisible due to transparency)

# # Get screen dimensions
# SCREEN_WIDTH = root.winfo_screenwidth()
# SCREEN_HEIGHT = root.winfo_screenheight()

# # Create a canvas that covers the entire screen
# canvas = tk.Canvas(root, width=SCREEN_WIDTH, height=SCREEN_HEIGHT, highlightthickness=0)
# canvas.pack()

# # Define custom rectangles (x1, y1, x2, y2)
# rectangles = [
#     (100, 100, 300, 250),  # Example: top-left (100, 100) to bottom-right (300, 250)
#     (400, 300, 500, 400),
#     (700, 200, 850, 400)
# ]

# # Draw rectangles on the canvas
# for rect in rectangles:
#     x1, y1, x2, y2 = rect
#     canvas.create_rectangle(x1, y1, x2, y2, outline='green', width=10)

# # Function to close the window
# def close_window(event=None):
#     root.quit()

# # Bind ESC key to close the window
# root.bind('<Escape>', close_window)

# # Optional: Add a way to close with a button (uncomment if desigreen)
# close_button = tk.Button(root, text="Close", command=root.quit)
# close_button.place(x=SCREEN_WIDTH-100, y=10)

# # Main loop to keep it running
# root.mainloop()

import tkinter as tk
from win32api import GetSystemMetrics
import win32gui
import win32con
from ultralytics import YOLO
import pyautogui
import numpy as np
import cv2
from PIL import Image, ImageTk, ImageFont, ImageDraw
from lib.manga_ocr import MangaOcr
import datetime
from langdetect import detect
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import keyboard
import uuid
import os

def setClickthrough(hwnd):
    print("setting window properties")
    try:
        styles = win32gui.GetWindowLong(hwnd, win32con.GWL_EXSTYLE)
        styles = win32con.WS_EX_TRANSPARENT
        win32gui.SetWindowLong(hwnd, win32con.GWL_EXSTYLE, styles)
        # win32gui.SetLayeredWindowAttributes(hwnd, 0, 255, win32con.LWA_ALPHA)
    except Exception as e:
        print(e)

# Dimensions
width = GetSystemMetrics(0) #self.winfo_screenwidth()
height = GetSystemMetrics(1) #self.winfo_screenheight()

root = tk.Tk()
root.geometry('%dx%d' % (width, height))
root.attributes('-fullscreen', False)
root.title("Applepie")
root.attributes("-topmost", 1)
root.attributes('-transparentcolor', 'green')
root.wm_attributes('-transparentcolor', 'green')
root.wm_attributes("-topmost", 1)
root.config(background='green') 
# root.attributes("-alpha", 0.75)
# root.overrideredirect(True)
root.wm_attributes('-fullscreen', True)
bg = tk.Canvas(root, width=width, height=height, background='green', bd=0, highlightthickness=0)

setClickthrough(bg.winfo_id())

# Load a model
detect_model = YOLO(r"pretrained_models/best1.pt")  # pretrained YOLO12n model
read_model = MangaOcr()
# read_model = ""
tokenizer_ja = AutoTokenizer.from_pretrained(r"D:\datatrain\ja_en")
model_ja = AutoModelForSeq2SeqLM.from_pretrained(r"D:\datatrain\ja_en")
tokenizer_zh = AutoTokenizer.from_pretrained(r"D:\datatrain\zh_en")
model_zh = AutoModelForSeq2SeqLM.from_pretrained(r"D:\datatrain\zh_en")
bbox_recs = []
temp_image = None
abcd = []

def translate_text(text, tokenizer, model):
    input_ids = tokenizer.encode(text, return_tensors="pt")
    outputs = model.generate(input_ids)
    decoded = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return decoded

def gettxtsize(text, font):
    left, top, right, bottom = font.getbbox(text)
    width = right - left
    height = bottom - top
    return width, height

def get_wrapped_text(text: str, font: ImageFont.ImageFont,
                     line_length: int):
        lines = ['']
        for word in text.split():
            line = f'{lines[-1]} {word}'.strip()
            if font.getlength(line) <= line_length:
                lines[-1] = line
            else:
                lines.append(word)
        return lines

def add_text_to_image(
    image_rgb: np.ndarray,
    label: str,
    top_left_xy: tuple = (0, 0),
    font_scale: float = 1,
    font_thickness: float = 1,
    font_face=cv2.FONT_HERSHEY_SIMPLEX,
    font_color_rgb: tuple = (0, 0, 255),
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

def replace_text(img):
    font = ImageFont.truetype(r"assets\arial.ttf", 13)
    read_text = read_model(img)
    if not read_text:
        return np.asarray(img)
    print(read_text)
    try:
        detectLanguage = detect(read_text)
    except:
        return np.asarray(img)
    print(read_text, '-->', detectLanguage)
    if detectLanguage == 'ja':
        translated_text = translate_text(read_text, tokenizer_ja, model_ja)
    elif 'zh' in detectLanguage:
        translated_text = translate_text(read_text, tokenizer_zh, model_zh)
    else: 
        translated_text = read_text
    print(read_text, "-->",translated_text)
    lines = get_wrapped_text(translated_text, font, img.size[0])
    lines = list(filter(None, lines))
    # max_width = 0
    # for line in lines:
    #     line_width = gettxtsize(line, font)[0]
    #     if line_width > max_width:
    #         max_width = line_width
    # new_w = int(max_width)
    # max_length = int((len(lines)+1)*gettxtsize(lines[0], font)[1])
    # new_h = max_length if max_length >img.size[1] else img.size[1]

    empty_img = 255 * np.ones((img.size[1], img.size[0], 3), dtype=np.uint8)
    return add_text_to_image(
        empty_img,
        '\n'.join(lines),
        top_left_xy=(0, 0),
    )
    # empty_img = Image.new("RGB", (new_w, new_h), (255,255,255))
    # new_img = ImageDraw.Draw(empty_img)
    # new_img.text((0, 0), '\n'.join(lines), fill="#000", font=font)
    # return empty_img

def clear_temp_dir():
    files = [x for x in os.listdir('temp')]
    for file in files:
        os.remove('temp/' + file)

def get_bounding_boxes():
    clear_temp_dir()
    global bbox_recs, abcd
    # bbox_recs.clear()
    # clear the canvas 
    bg.delete('all') 
    bg.update()
    abcde = []
    if not bg.winfo_children():
        img = pyautogui.screenshot()
        # img.show()
        # Convert the screenshot to a numpy array
        frame = np.array(img)

        # Convert it from BGR(Blue, Green, Red) to
        # RGB(Red, Green, Blue)
        detect_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        results = detect_model.predict(source=detect_frame, imgsz=1280, conf=0.1, classes = [0], agnostic_nms=True) #
        bounding_boxes = results[0].boxes.xyxy.tolist()
        if bounding_boxes and len(bounding_boxes) != 0:
            bounding_boxes.sort(key=lambda x: x[1])
            for coor in bounding_boxes:
                coor = list(map(int, coor)) #Convert float to int 
                crop_image = img.crop((coor[0],coor[1], coor[2],coor[3]))
                tempo_image = replace_text(crop_image)
                # temp_filename = 'temp/' + str(uuid.uuid4()) + '.png'
                # cv2.imwrite(temp_filename, tempo_image)
                # temp_image = ImageTk.PhotoImage(Image.open(temp_filename))
                temp_image = ImageTk.PhotoImage(Image.fromarray(tempo_image))
                # tempo_image.show()
                # print(read_model(img.crop((coor[0],coor[1], coor[2],coor[3]))))
                bbox_recs.append(temp_image)
                abcd.append(tempo_image)
                bg.create_image(coor[0], coor[1], image = temp_image, anchor=tk.NW)
                bg.update()
            bg.pack()
        # root.after(10000, get_bounding_boxes)  # reschedule event in 2 seconds

root.after(1, get_bounding_boxes)

def close_window(event=None):
    root.quit()


root.bind('<Escape>', close_window)
keyboard.add_hotkey('ctrl+a', get_bounding_boxes)




# frame = ImageTk.PhotoImage(file="result.jpg")
# bg.create_image(1920/2, 1080/2, image=frame)
# bg.pack()
root.mainloop()