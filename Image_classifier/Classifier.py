import os
import numpy as np
import tkinter as tk
from tkinter import *
import tkinter.filedialog
from PIL import Image, ImageTk
from tensorflow.keras.applications import vgg16
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.applications.vgg16 import decode_predictions

root = tk.Tk()


def display_image():
    global image, image_data

    # claer the canvas
    for image_display in frame.winfo_children():
        image_display.destroy()

    # imput the image to the canvas
    image_data = tk.filedialog.askopenfilename(initialdir=os.getcwd(), title="Choose an image",
                                               filetypes=(("all files", "*.*"), ("png files", "*.png")))
    # minimum width of the canvas image
    basewidth = 300

    # resize the image to fit accordingly to the canvas
    image = Image.open(image_data)
    width_percent = (basewidth / float(image.size[0]))
    height = int((float(image.size[1]) * float(width_percent)))
    image = image.resize((basewidth, height), Image.ANTIALIAS)
    image = ImageTk.PhotoImage(image)
    file_name = image_data.split('/')
    panel = tk.Label(frame, text=str(file_name[len(file_name) - 1]).upper()).pack()
    panel_image = tk.Label(frame, image=image).pack()


def classify_image():
    original = Image.open(image_data)
    original = original.resize((224, 224), Image.ANTIALIAS)

    numpy_image = img_to_array(original)
    image_batch = np.expand_dims(numpy_image, axis=0)

    processed_image = vgg16.preprocess_input(image_batch.copy())
    predictions = vgg_model.predict(processed_image)

    label = decode_predictions(predictions)
    table = tk.Label(frame, text="Top image class predictions and confidences\n").pack()

    for i in range(0, len(label[0])):
         result = tk.Label(frame,
                    text= str(label[0][i][1]).upper() + ': ' +
                           str(round(float(label[0][i][2])*100, 3)) + '%').pack()


root.title('Image Classifier')
title = Label(root, text="Image Classifier", padx=25, pady=5, font=("", 15)).pack()

canvas = Canvas(root, height=700, width=700, bg='grey')
canvas.pack()

frame = Frame(root, bg="white")
frame.place(relwidth=0.8, relheight=0.8, relx=0.1, rely=0.1)

open_image = Button(root, text='Choose Image', padx=35, pady=10, fg="white", bg="grey", command=display_image)
open_image.pack(side=LEFT)

class_image = Button(root, text='Classify Image', padx=35, pady=10, fg="white", bg="grey", command=classify_image)
class_image.pack(side=RIGHT)

vgg_model = vgg16.VGG16(weights='imagenet')

root.resizable(False, False)
root.mainloop()