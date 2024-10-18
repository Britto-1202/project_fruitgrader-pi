import cv2
import numpy as np
import tensorflow as tf
import pyttsx3
import board
import digitalio
import adafruit_character_lcd.character_lcd as characterlcd
lcd_columns = 16
lcd_rows = 2
lcd_rs = digitalio.DigitalInOut(board.D26)
lcd_en = digitalio.DigitalInOut(board.D19)
lcd_d4 = digitalio.DigitalInOut(board.D13)
lcd_d5 = digitalio.DigitalInOut(board.D6)
lcd_d6 = digitalio.DigitalInOut(board.D5)
lcd_d7 = digitalio.DigitalInOut(board.D11)
lcd = characterlcd.Character_LCD_Mono(lcd_rs, lcd_en, lcd_d4, lcd_d5, lcd_d6, lcd_d7, lcd_columns, lcd_rows)

# Load the pre-trained ML model
model = tf.keras.models.load_model('/home/echo/project/env/lib/python3.9/site-packages/fruits_datas.h5')

# Initialize the text-to-speech engine
engine = pyttsx3.init()

# Initialize the LCD
#lcd = CharLCD('PCF8574', 0x27)

# Start capturing video from webcam
cap = cv2.VideoCapture(0)

# Set the frame size
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

# Define the crop region for the fruit


while True:
    # Read a frame from the video stream
    ret, frame = cap.read()

    # Display the original frame
    cv2.imshow('Fruit Detector', frame)

    # Wait for key press
    key = cv2.waitKey(1)
    if key == ord('q'):
        break
    elif key == ord(' '):
        # Capture the image and save it to a file
        fruit = frame
        cv2.imwrite('fruit.jpg', fruit)

        # Preprocess the image for the ML model
        img = tf.keras.preprocessing.image.img_to_array(fruit)
        img = tf.keras.preprocessing.image.smart_resize(img, (224, 224))
        img = tf.keras.applications.mobilenet_v2.preprocess_input(img)

        # Pass the image to the ML model and get the prediction
        preds = model.predict(np.expand_dims(img, axis=0))
        print(preds)

        # Define a dictionary to map indices to fruit labels
        label_dict = {0: "fresh apple", 1: "fresh banana", 2: "fresh orange",
                      3: "rotten apple", 4: "rotten banana", 5: "rotten orange"}

        # Get the index of the maximum value in the preds array
        pred_index = np.argmax(preds)
        print(pred_index)
        # Check if the predicted class is above the threshold
        if preds[0][pred_index] > 0.5:
            # Get the corresponding fruit label from the dictionary
            label = label_dict[pred_index]
            print("The fruit is a "+ label+".")
            # Speak the label
            engine.say("The fruit is a "+ label+".")
            engine.runAndWait()
            # Display the label on the LCD
            print(label[:lcd_columns])
            print((preds[0][pred_index]*100))
            lcd.clear()
            lcd.message='Fruit: ' + label[:lcd_columns] + '\nConfidence: {:.2f}%'.format(preds[0][pred_index]*100)
        else:
            print("Please show a valid image.")
            # Speak an error message
            engine.say("Please show a valid image.")
            engine.runAndWait()
            # Display an error message on the LCD
            lcd.clear()
            lcd.message='Error: Invalid\nImage'

# Release the capture and destroy all windows
cap.release()
cv2.destroyAllWindows()