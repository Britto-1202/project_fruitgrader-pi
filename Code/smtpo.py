#!/usr/bin/env python
import cv2
import os
import  RPi.GPIO as GPIO
import time
import zipfile
import smtplib
from email.message import EmailMessage
import tensorflow as tf
import numpy as np
import csv
import pyttsx3
import board
import digitalio
import LiquidCrystal_I2C
from time import *

mylcd =  LiquidCrystal_I2C.lcd()

mylcd.lcd_display_string("Hello!", 1)
# Initialize the text-to-speech engine
engine = pyttsx3.init()
# Load the trained ML model
model = tf.keras.models.load_model(
    '/home/echo/project/env/lib/python3.9/site-packages/fruits_datas.h5')

# Set up the email parameters
from_email = 'codeinechoniner@gmail.com'
from_password = 'yhqjkahoevcixcth'
to_email = 'jeromgladsun@gmail.com'

# Set up the folder for saving the images
image_folder = 'fruit_images'
if not os.path.exists(image_folder):
    os.mkdir(image_folder)
# Delete all files in the image folder
filelist = [f for f in os.listdir(image_folder)]
for f in filelist:
    os.remove(os.path.join(image_folder, f))


# Set up the CSV file for saving the results
csv_filename = 'iruit_results.csv'
with open(csv_filename, mode='w', newline='') as csv_file:
    writer = csv.writer(csv_file)
    writer.writerow(['Image File', 'Label', 'Fruit_Name', 'Confidence'])

# Set up the counters for fresh and rotten fruits
fresh_count = 0
rotten_count = 0

# Start capturing video from webcam
cap = cv2.VideoCapture(0)

# Set the frame size
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

GPIO.setmode(GPIO.BCM)
GPIO.setup(18, GPIO.IN, pull_up_down=GPIO.PUD_UP)
GPIO.setmode(GPIO.BCM)
GPIO.setup(19, GPIO.IN, pull_up_down=GPIO.PUD_UP)
    

while True:
    # Read a frame from the video stream
    ret, frame = cap.read()

    # Display the original frame
    cv2.imshow('Fruit Detector', frame)

    # Wait for key press
    key = cv2.waitKey(1)
    if not GPIO.input(19):
        # Create a zip archive of the image folder
        zip_filename = 'fruit_images.zip'
        with zipfile.ZipFile(zip_filename, 'w', zipfile.ZIP_DEFLATED) as zip_file:
            for root, dirs, files in os.walk(image_folder):
                for file in files:
                    zip_file.write(os.path.join(root, file))

        # Send an email with the results and the image folder zip archive
        msg = EmailMessage()
        msg['Subject'] = 'Fruit Detection Results'
        msg['From'] = from_email
        msg['To'] = to_email
        msg.set_content(
            f'Fresh fruits: {fresh_count}, Rotten fruits: {rotten_count}')

        with open(csv_filename, 'rb') as csv_file:
            csv_data = csv_file.read()
            msg.add_attachment(csv_data, maintype='text',
                               subtype='csv', filename=csv_filename)

        with open(zip_filename, 'rb') as zip_file:
            zip_data = zip_file.read()
            msg.add_attachment(zip_data, maintype='application',
                               subtype='zip', filename=zip_filename)

        with smtplib.SMTP_SSL('smtp.gmail.com', 465) as smtp:
            smtp.login(from_email, from_password)
            smtp.send_message(msg)

        break
    elif not GPIO.input(18):
        # Capture the image and save it to a file
        fruit = frame
        image_filename = os.path.join(
            image_folder, f'fruit_{len(os.listdir(image_folder))+1}.jpg')
        cv2.imwrite(image_filename, fruit)

        # Preprocess the image for the ML model
        img = tf.keras.preprocessing.image.img_to_array(fruit)
        img = tf.keras.preprocessing.image.smart_resize(img, (224, 224))
        img = tf.keras.applications.mobilenet_v2.preprocess_input(img)

        # Pass the image to the ML model and get the prediction
        preds = model.predict(np.expand_dims(img, axis=0))
        label_dict = {0: "fresh apple", 1: "fresh banana", 2: "fresh orange",
                      3: "rotten apple", 4: "rotten banana", 5: "rotten orange"}
        pred_index = np.argmax(preds)
        print(pred_index)
        # Define
        fruit_name = label_dict[pred_index]
        if preds[0][pred_index] > 0.5:
            if 'fresh' in fruit_name:
                fruit_label = 'fresh'
                fresh_count += 1
                # Speak the label
                confidence = (preds[0][pred_index]*100)
                print("The fruit is a " + fruit_name+".",confidence)
                engine.say("The fruit is a " + fruit_name+".")
                engine.runAndWait()
                mylcd.lcd_clear()
                mylcd.lcd_display_string(fruit_name, 1)
                mylcd.lcd_display_string('Percent: {:.2f}%'.format(preds[0][pred_index]*100),2)

            elif 'rotten' in fruit_name:
                fruit_label = 'rotten'
                rotten_count += 1
                
                # Speak the label
                confidence = (preds[0][pred_index]*100)
                print("The fruit is a " + fruit_name+".",confidence)
                engine.say("The fruit is a " + fruit_name+".")
                engine.runAndWait()
                mylcd.lcd_clear()
                mylcd.lcd_display_string(fruit_name, 1)
                mylcd.lcd_display_string('Percent: {:.2f}%'.format(preds[0][pred_index]*100),2)

        else:
            print("Please show a valid image.")
            # Speak an error message
            engine.say("Please show a valid image.")
            engine.runAndWait()
            # Display an error messa
            mylcd.lcd_clear()
            mylcd.lcd_display_string('Error: Invalid Image',1)
        # Write the results to the CSV file
        with open(csv_filename, mode='a', newline='') as csv_file:
            writer = csv.writer(csv_file)
            writer.writerow(
                [image_filename, fruit_label, fruit_name, confidence])

        cv2.imshow('Fruit Detector', frame)
cap.release()
cv2.destroyAllWindows()


