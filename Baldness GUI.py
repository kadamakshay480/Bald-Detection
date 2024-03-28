import tkinter as tk
from tkinter import filedialog
from tkinter import *
from PIL import Image,ImageTk
import numpy as np
import os
import cv2

# Loading the model
from keras.models import load_model

# Initializing the GUI
top = tk.Tk()
top.geometry('800x600')
top.title('Baldness Detector')
top.configure(background='#CDCDCD')

# Initializing the label
result_label = Label(top, background='#CDCDCD', font=('Arial', 20))

# Loading the trained model
model = load_model('bald_classifier_model.h5')

# Defining detect function which detects whether a person is bald or not using the model
def detect_baldness(file_path):
    try:
        image = Image.open(file_path)
        image = image.resize((64, 64))
        image = np.array(image)
        image = np.expand_dims(image, axis=0)
        image = image / 255.0  # Normalize pixel values to [0, 1]

        pred = model.predict(image)
        if pred[0][0] > 0.7:
            result_label.configure(foreground='#011638', text='Not Bald')
        else:
            result_label.configure(foreground='#011638', text='Bald')
    except Exception as e:
        print(e)
        result_label.configure(foreground='#011638', text='Error processing image.')

# Defining upload image function
def upload_image():
    try:
        file_path = filedialog.askopenfilename()
        detect_baldness(file_path)
    except:
        pass

# Create and pack GUI elements
upload_button = Button(top, text='Upload an Image', command=upload_image, padx=10, pady=5)
upload_button.configure(background='#364156', foreground='white', font=('Arial', 10, 'bold'))
upload_button.pack(side='bottom', pady=50)

result_label.pack(side='bottom', expand=True)

heading = Label(top, text='Baldness Detector', pady=20, font=('Arial', 20, 'bold'))
heading.configure(background='#CDCDCD', foreground='#364156')
heading.pack()

top.mainloop()
