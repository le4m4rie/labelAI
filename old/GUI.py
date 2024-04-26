import tkinter as tk
import cv2 as cv
from LabelBot import LabelBot
from EasyReader import EasyReader
from Detector import Detector
import re

labelBot = LabelBot()
easyReader = EasyReader()
detector = Detector()

def detect(img_path: str):
   img = cv.imread(img_path)
   if img is None:
      return "error: file path not found"
   objects = detector.detect_label(img)
   if len(objects) == 0:
      return "AI RESULT: label not found"
   else:
      for (x, y, w, h) in objects:
         cv.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)
         roi = img[y:y+h, x:x+w]
         results = easyReader.reader.readtext(roi, allowlist='0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ')
         numbers = []
         for detection in results:
            if detection[2] > 0.8:
                numbers.append(detection[1])
         my_string = "".join(numbers)
         for i in range(len(my_string)):
            if my_string[i] == 'O':
                my_string = my_string[:i] + '0' + my_string[i+1:]
            elif my_string[i] == 'I':
                my_string = my_string[:i] + '1' + my_string[i+1:]
         pattern = r"[0-9]{4}[0-9A-Z]{7}"
         matches = re.findall(pattern, my_string)
         if matches:
            return "AI RESULT: item number detected: " + matches[0]
         else:
            return "AI RESULT: item number not or not fully detected"

def get_input():
   input_value = entry.get()
   output_value = detect(input_value)
   output_field.delete('1.0', tk.END)
   output_field.insert(tk.END, output_value)
   entry.delete(0, tk.END)

root = tk.Tk()
root.minsize(600,200)
root.title("Label and item number detector")

entry = tk.Entry(root, font=("Arial", 16))
entry.pack(padx=20, pady=20)

button = tk.Button(root, text="check!", font=("Arial", 16), command=get_input)
button.pack(padx=20, pady=20)

output_field = tk.Text(root, font=("Arial", 16), height=1)
output_field.pack(padx=20, pady=20)

root.mainloop()
