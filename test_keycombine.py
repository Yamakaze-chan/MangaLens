import keyboard
import tkinter as tk

def a():
    print('a')
root = tk.Tk()
keyboard.add_hotkey('ctrl+alt', a)
root.mainloop()