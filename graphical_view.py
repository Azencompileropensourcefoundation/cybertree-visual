import json
import sys
import tkinter as tk
from tkinter import ttk
import numpy as np
import cv2
from PIL import Image, ImageTk
import tensorflow as tf
import requests
import math
import datetime
import pytz
import os

# Model and dependencies
sys.path.append(os.path.join(os.getcwd(), 'instructorship-training'))

from nethumandetector_IO import detect_and_classify_face, query_human_behavior, analyze_and_respond_to_behavior

class ObjectDetectionApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Sniper and Object Detection Interface")
        self.root.geometry("1920x1080")
        self.is_fullscreen = True
        self.is_wifi_on = False
        self.is_gray_area_visible = False
        
        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            print("Error: Could not open webcam.")
            self.root.quit()
            return

        # Load MobileNet model (nethumandetector.h5)
        model_path = 'trained-trainer/nethumandetector.h5'
        if not os.path.exists(model_path):
            print(f"Error: Model file '{model_path}' not found.")
            self.root.quit()
            return
        
        try:
            self.face_model = tf.keras.models.load_model(model_path)
        except Exception as e:
            print(f"Error loading model: {e}")
            self.root.quit()
            return

        # Create Sniper Bar
        self.sniper_bar = tk.Canvas(self.root, width=800, height=50, bg='black')
        self.sniper_bar.pack(side=tk.TOP, pady=20)

        self.control_frame = ttk.Frame(self.root, padding=10, relief="sunken")
        self.control_frame.pack(side=tk.RIGHT, fill=tk.Y, padx=10)

        self.toggle_wifi_button = ttk.Button(self.control_frame, text="Toggle Wi-Fi", command=self.toggle_wifi)
        self.toggle_wifi_button.pack(side=tk.TOP, padx=10, pady=10)

        self.toggle_grayscale_button = ttk.Button(self.control_frame, text="Toggle Gray Area", command=self.toggle_gray_area)
        self.toggle_grayscale_button.pack(side=tk.TOP, padx=10, pady=10)

        self.compass_label = ttk.Label(self.control_frame, text="Compass: N", foreground="white", background="black")
        self.compass_label.pack(side=tk.TOP, pady=10)

        self.time_label = ttk.Label(self.control_frame, text="Time: Loading...", foreground="white", background="black")
        self.time_label.pack(side=tk.TOP, pady=10)

        self.current_location = self.get_ip_location() if self.is_wifi_on else (0, 0)
        self.target_location = self.current_location
        self.compass_direction = "N"
        self.update_sniper_bar("Safe", "Human")

        self.canvas = tk.Canvas(self.root, bg='black', bd=0, highlightthickness=0)
        self.canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        self.frame_skip_count = 0  # Variable to skip frames for performance
        self.update_frame()
        self.root.bind("<F11>", self.toggle_fullscreen)
        self.update_time()

    def get_ip_location(self):
        if not self.is_wifi_on:
            return (0, 0)
        try:
            ip_info = requests.get("http://ipinfo.io/json").json()
            loc = ip_info['loc'].split(',')
            lat, lon = float(loc[0]), float(loc[1])
            return (lat, lon)
        except requests.exceptions.RequestException as e:
            print("Error getting IP location:", e)
            return (0, 0)

    def update_frame(self):
        ret, frame = self.cap.read()
        if not ret:
            print("Error: Failed to capture frame.")
            frame = np.zeros((1080, 1920, 3), dtype=np.uint8)

        # Downscale the frame for faster processing
        frame = cv2.resize(frame, (1920, 1080))

        frame = self.apply_night_vision(frame)

        # Skip frames to reduce processing load (adjust every 5th frame for example)
        if self.frame_skip_count % 5 == 0:
            try:
                # Use the MobileNet model to detect and classify face and human behavior
                frame = detect_and_classify_face(self.face_model, frame)
                # Query the human behavior from the external source
                human_data = query_human_behavior(frame)
                if human_data:
                    analyze_and_respond_to_behavior(human_data)
            except Exception as e:
                print(f"Error during face detection and classification: {e}")

        # Increment frame skip counter
        self.frame_skip_count += 1

        # Update canvas with the captured frame
        frame_tk = self.convert_to_tkinter_image(frame)
        self.canvas.create_image(0, 0, anchor=tk.NW, image=frame_tk)
        self.canvas.image = frame_tk

        self.draw_crosshair_and_compass()
        self.update_compass()

        self.root.after(10, self.update_frame)  # Continue updating frames

    def convert_to_tkinter_image(self, frame):
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_tk = ImageTk.PhotoImage(image=Image.fromarray(frame_rgb))
        return frame_tk

    def apply_night_vision(self, frame):
        green_tinted_frame = frame.copy()
        green_tinted_frame[:, :, 1] = green_tinted_frame[:, :, 1] * 1.5
        contrast_frame = cv2.convertScaleAbs(green_tinted_frame, alpha=1.5, beta=30)
        contrast_frame = np.clip(contrast_frame, 0, 255)
        return contrast_frame

    def draw_crosshair_and_compass(self):
        self.canvas.create_line(self.root.winfo_width() // 2, 0, self.root.winfo_width() // 2, self.root.winfo_height(), fill="white", width=2)
        self.canvas.create_line(0, self.root.winfo_height() // 2, self.root.winfo_width(), self.root.winfo_height() // 2, fill="white", width=2)
        self.draw_dynamic_compass()

    def draw_dynamic_compass(self):
        angle = self.calculate_compass_angle(self.current_location, self.target_location)
        self.canvas.create_arc(self.root.winfo_width() // 2 - 50, self.root.winfo_height() // 2 - 50, self.root.winfo_width() // 2 + 50, self.root.winfo_height() // 2 + 50,
                               start=angle, extent=180, outline="white", width=5)

    def calculate_compass_angle(self, current_loc, target_loc):
        lat1, lon1 = current_loc
        lat2, lon2 = target_loc

        lat1 = math.radians(lat1)
        lon1 = math.radians(lon1)
        lat2 = math.radians(lat2)
        lon2 = math.radians(lon2)

        delta_lon = lon2 - lon1
        x = math.sin(delta_lon) * math.cos(lat2)
        y = math.cos(lat1) * math.sin(lat2) - math.sin(lat1) * math.cos(lat2) * math.cos(delta_lon)
        bearing = math.atan2(x, y)

        bearing = math.degrees(bearing)
        bearing = (bearing + 360) % 360
        return bearing

    def update_sniper_bar(self, object_class, category):
        color = "green"
        if object_class == "Aggressive":
            color = "red"
        elif object_class == "Robot":
            color = "purple"
        self.sniper_bar.config(bg=color)

    def update_compass(self):
        self.compass_label.config(text=f"Compass: {self.compass_direction}")

    def toggle_wifi(self):
        self.is_wifi_on = not self.is_wifi_on
        if self.is_wifi_on:
            self.current_location = self.get_ip_location()
            self.toggle_wifi_button.config(text="Turn Wi-Fi Off")
        else:
            self.current_location = (0, 0)
            self.toggle_wifi_button.config(text="Turn Wi-Fi On")

    def toggle_gray_area(self):
        self.is_gray_area_visible = not self.is_gray_area_visible
        self.canvas.config(bg="gray" if self.is_gray_area_visible else "black")

    def toggle_fullscreen(self, event=None):
        self.is_fullscreen = not self.is_fullscreen
        if self.is_fullscreen:
            self.root.attributes("-fullscreen", True)
            self.root.config(bg="black")
        else:
            self.root.attributes("-fullscreen", False)
            self.root.config(bg="white")

    def update_time(self):
        local_time = datetime.datetime.now().strftime("%H:%M:%S")
        gmt_time = datetime.datetime.now(pytz.utc).strftime("%H:%M:%S")
        self.time_label.config(text=f"Local Time: {local_time} | GMT: {gmt_time}")
        self.root.after(1000, self.update_time)


if __name__ == "__main__":
    root = tk.Tk()
    app = ObjectDetectionApp(root)
    root.mainloop()
