import tkinter as tk
import re
from win32api import GetSystemMetrics

def measure_translate_area():
    """Hộp thoại bán trong suốt hỗ trợ kéo thả vùng dịch trên màn hình."""
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


def gettxtsize(text, font):
    """Tính toán chiều rộng và cao của đoạn text dựa vào Font."""
    left, top, right, bottom = font.getbbox(text)
    width = right - left
    height = bottom - top
    return width, height


def text_wrap(text, font, max_width):
    """Tự động xuống dòng từ ngữ để vừa vặn vào khung ảnh dịch."""
    lines = []
    if gettxtsize(text, font)[0] <= max_width - 5:
        lines.append(text)
    else:
        try:
            words = re.split(r'(\W+)', text)
            words = [w for w in words if w]
            i = 0
            while i < len(words):
                line = ''
                while i < len(words) and gettxtsize(line + words[i], font)[0] <= max_width - 5:
                    line = line + words[i]
                    i += 1

                while i < len(words) and re.match(r'^\W+$', words[i]) and not words[i].strip() == '':
                    line = line + words[i]
                    i += 1

                if not line:
                    line = words[i]
                    i += 1
                if line.strip() != '':
                    if re.fullmatch(r'[\W_]+', line):
                        if lines:
                            lines[-1] += line
                    else:
                        lines.append(line)
        except Exception as e:
            print(f"Lỗi khi ngắt dòng chữ: {e}")
        finally:
            return lines