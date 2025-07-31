import tkinter as tk
from tkinter import ttk, messagebox
import cv2
from PIL import Image, ImageTk
import os
import numpy as np
import threading
import time
import tensorflow as tf
from tensorflow.keras.models import load_model

class FaceAuthSystem:
    def __init__(self, root):
        self.root = root
        self.root.title("Face Recognition Authentication System")
        self.root.geometry("800x600")
        self.root.configure(bg='#f0f0f0')
        
        # Create users directory if it doesn't exist
        self.users_dir = "../users"
        if not os.path.exists(self.users_dir):
            os.makedirs(self.users_dir)
        
        # Initialize camera
        self.cap = None
        self.camera_running = False
        self.current_frame = None
        
        # Initialize face detection
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        self.face_detected = False
        self.face_confidence_threshold = 0.6
        
        # Load anti-spoofing model
        try:
            self.model = load_model('../model/anti_spoofing_model_upgraded1.keras')
            print("Anti-spoofing model loaded successfully")
        except Exception as e:
            print(f"Error loading model: {str(e)}")
            self.model = None
        
        # Initialize pages
        self.current_page = None
        self.show_home_page()
    
    def clear_window(self):
        """Clear all widgets from the window"""
        for widget in self.root.winfo_children():
            widget.destroy()
    
    def show_home_page(self):
        """Display the home page with Sign Up and Log In buttons"""
        self.clear_window()
        self.stop_camera()
        self.current_page = "home"
        
        title_label = tk.Label(self.root, text="Face Recognition Authentication", 
                              font=("Arial", 24, "bold"), bg='#f0f0f0', fg='#333')
        title_label.pack(pady=50)
        
        button_frame = tk.Frame(self.root, bg='#f0f0f0')
        button_frame.pack(pady=50)
        
        signup_btn = tk.Button(button_frame, text="Sign Up", font=("Arial", 16),
                              bg='#4CAF50', fg='white', padx=30, pady=10,
                              command=self.show_signup_page, cursor='hand2')
        signup_btn.pack(side=tk.LEFT, padx=20)
        
        login_btn = tk.Button(button_frame, text="Log In", font=("Arial", 16),
                             bg='#2196F3', fg='white', padx=30, pady=10,
                             command=self.show_login_page, cursor='hand2')
        login_btn.pack(side=tk.LEFT, padx=20)
    
    def show_signup_page(self):
        """Display the sign-up page"""
        self.clear_window()
        self.current_page = "signup"
        
        title_label = tk.Label(self.root, text="Sign Up", font=("Arial", 20, "bold"),
                              bg='#f0f0f0', fg='#333')
        title_label.pack(pady=20)
        
        main_frame = tk.Frame(self.root, bg='#f0f0f0')
        main_frame.pack(expand=True, fill='both', padx=20, pady=20)
        
        left_frame = tk.Frame(main_frame, bg='#f0f0f0')
        left_frame.pack(side=tk.LEFT, fill='both', expand=True)
        
        self.camera_label = tk.Label(left_frame, text="Camera Feed", 
                                    bg='black', fg='white', font=("Arial", 12))
        self.camera_label.pack(pady=10)
        
        right_frame = tk.Frame(main_frame, bg='#f0f0f0', width=250)
        right_frame.pack(side=tk.RIGHT, fill='y', padx=(20, 0))
        right_frame.pack_propagate(False)
        
        tk.Label(right_frame, text="Enter your name:", font=("Arial", 12),
                bg='#f0f0f0').pack(pady=(20, 5))
        
        self.name_entry = tk.Entry(right_frame, font=("Arial", 12), width=20)
        self.name_entry.pack(pady=5)
        
        self.capture_btn = tk.Button(right_frame, text="Capture", font=("Arial", 12),
                               bg='#FF9800', fg='white', padx=20, pady=5,
                               command=self.capture_face, cursor='hand2', state='disabled')
        self.capture_btn.pack(pady=10)
        
        ok_btn = tk.Button(right_frame, text="OK", font=("Arial", 12),
                          bg='#4CAF50', fg='white', padx=20, pady=5,
                          command=self.save_user, cursor='hand2')
        ok_btn.pack(pady=5)
        
        back_btn = tk.Button(right_frame, text="Back", font=("Arial", 12),
                            bg='#757575', fg='white', padx=20, pady=5,
                            command=self.show_home_page, cursor='hand2')
        back_btn.pack(pady=5)
        
        self.face_status_label = tk.Label(right_frame, text="", font=("Arial", 10),
                                         bg='#f0f0f0', wraplength=200)
        self.face_status_label.pack(pady=5)
        
        self.status_label = tk.Label(right_frame, text="", font=("Arial", 10),
                                    bg='#f0f0f0', wraplength=200)
        self.status_label.pack(pady=10)
        
        self.captured_image = None
        self.captured_face = None  # Store cropped face
        self.start_camera()
    
    def show_login_page(self):
        """Display the login page"""
        self.clear_window()
        self.current_page = "login"
        
        title_label = tk.Label(self.root, text="Log In", font=("Arial", 20, "bold"),
                              bg='#f0f0f0', fg='#333')
        title_label.pack(pady=20)
        
        main_frame = tk.Frame(self.root, bg='#f0f0f0')
        main_frame.pack(expand=True, fill='both', padx=20, pady=20)
        
        left_frame = tk.Frame(main_frame, bg='#f0f0f0')
        left_frame.pack(side=tk.LEFT, fill='both', expand=True)
        
        self.camera_label = tk.Label(left_frame, text="Camera Feed",
                                    bg='black', fg='white', font=("Arial", 12))
        self.camera_label.pack(pady=10)
        
        right_frame = tk.Frame(main_frame, bg='#f0f0f0', width=250)
        right_frame.pack(side=tk.RIGHT, fill='y', padx=(20, 0))
        right_frame.pack_propagate(False)
        
        tk.Label(right_frame, text="Enter your name:", font=("Arial", 12),
                bg='#f0f0f0').pack(pady=(20, 5))
        
        self.login_name_entry = tk.Entry(right_frame, font=("Arial", 12), width=20)
        self.login_name_entry.pack(pady=5)
        
        verify_btn = tk.Button(right_frame, text="Verify", font=("Arial", 12),
                              bg='#2196F3', fg='white', padx=20, pady=5,
                              command=self.verify_face, cursor='hand2')
        verify_btn.pack(pady=10)
        
        back_btn = tk.Button(right_frame, text="Back", font=("Arial", 12),
                            bg='#757575', fg='white', padx=20, pady=5,
                            command=self.show_home_page, cursor='hand2')
        back_btn.pack(pady=5)
        
        self.login_status_label = tk.Label(right_frame, text="", font=("Arial", 10),
                                          bg='#f0f0f0', wraplength=200)
        self.login_status_label.pack(pady=10)
        
        self.captured_face = None  # Store cropped face for login
        self.start_camera()
    
    def preprocess_image(self, image, img_size=(112, 112)):
        """Preprocess image for model input"""
        if image is None:
            return None
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, img_size)
        image = image.astype(np.float32)
        image = np.clip(image, 0, 255)
        image = tf.keras.applications.efficientnet_v2.preprocess_input(image)
        image = np.expand_dims(image, axis=0)  # Add batch dimension
        return image
    
    def crop_face(self, frame):
        """Crop the largest detected face from the frame"""
        try:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = self.face_cascade.detectMultiScale(
                gray,
                scaleFactor=1.1,
                minNeighbors=5,
                minSize=(50, 50),
                flags=cv2.CASCADE_SCALE_IMAGE
            )
            
            if len(faces) > 0:
                # Get the largest face
                largest_face = max(faces, key=lambda face: face[2] * face[3])
                x, y, w, h = largest_face
                # Add padding to ensure the face is fully captured (e.g., 10% of face size)
                padding = int(max(w, h) * 0.1)
                x = max(0, x - padding)
                y = max(0, y - padding)
                w = min(frame.shape[1] - x, w + 2 * padding)
                h = min(frame.shape[0] - y, h + 2 * padding)
                # Crop the face
                cropped_face = frame[y:y+h, x:x+w]
                return cropped_face
            return None
        except Exception as e:
            print(f"Error in face cropping: {str(e)}")
            return None
    
    def start_camera(self):
        """Start the camera feed"""
        if not self.camera_running:
            self.cap = cv2.VideoCapture(0)
            self.camera_running = True
            self.update_camera_feed()
    
    def stop_camera(self):
        """Stop the camera feed"""
        self.camera_running = False
        if self.cap:
            self.cap.release()
            self.cap = None
    
    def update_camera_feed(self):
        """Update the camera feed display with horizontal flip"""
        if self.camera_running and self.cap and self.cap.isOpened():
            ret, frame = self.cap.read()
            if ret:
                # Flip the frame horizontally to mirror the user's movements
                frame = cv2.flip(frame, 1)
                self.current_frame = frame.copy()
                
                if self.current_page == "signup":
                    self.detect_face_in_frame(frame)
                
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame_resized = cv2.resize(frame_rgb, (480, 360))
                img = Image.fromarray(frame_resized)
                photo = ImageTk.PhotoImage(img)
                if hasattr(self, 'camera_label'):
                    self.camera_label.configure(image=photo)
                    self.camera_label.image = photo
            
            if self.camera_running:
                self.root.after(30, self.update_camera_feed)
    
    def detect_face_in_frame(self, frame):
        """Detect faces in the current frame and update UI accordingly"""
        try:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = self.face_cascade.detectMultiScale(
                gray,
                scaleFactor=1.1,
                minNeighbors=5,
                minSize=(50, 50),
                flags=cv2.CASCADE_SCALE_IMAGE
            )
            
            face_detected = len(faces) > 0
            
            if face_detected:
                largest_face = max(faces, key=lambda face: face[2] * face[3])
                x, y, w, h = largest_face
                face_area = w * h
                frame_area = frame.shape[0] * frame.shape[1]
                face_ratio = face_area / frame_area
                
                good_detection = face_ratio > 0.02
                
                if good_detection:
                    self.face_detected = True
                    self.face_status_label.configure(text="✓ Face detected - Ready to capture", fg='green')
                    if hasattr(self, 'capture_btn'):
                        self.capture_btn.configure(state='normal', bg='#FF9800')
                else:
                    self.face_detected = False
                    self.face_status_label.configure(text="⚠ Face too small - Move closer", fg='orange')
                    if hasattr(self, 'capture_btn'):
                        self.capture_btn.configure(state='disabled', bg='#CCCCCC')
            else:
                self.face_detected = False
                self.face_status_label.configure(text="✗ No face detected", fg='red')
                if hasattr(self, 'capture_btn'):
                    self.capture_btn.configure(state='disabled', bg='#CCCCCC')
                    
        except Exception as e:
            self.face_detected = False
            if hasattr(self, 'face_status_label'):
                self.face_status_label.configure(text="Error in face detection", fg='red')
    
    def capture_face(self):
        """Capture and crop the face from the current frame"""
        if not self.face_detected:
            self.status_label.configure(text="Error: No face detected in frame", fg='red')
            return
            
        if self.current_frame is not None:
            # Crop the face from the current frame
            cropped_face = self.crop_face(self.current_frame)
            if cropped_face is not None:
                self.captured_image = self.current_frame.copy()  # Keep full frame for display
                self.captured_face = cropped_face  # Store cropped face
                self.status_label.configure(text="Face captured successfully!", fg='green')
            else:
                self.status_label.configure(text="Error: Failed to crop face", fg='red')
        else:
            self.status_label.configure(text="Error: No camera feed available", fg='red')
    
    def save_user(self):
        """Save the cropped face and name"""
        name = self.name_entry.get().strip()
        
        if not name:
            self.status_label.configure(text="Error: Please enter a name", fg='red')
            return
        
        if self.captured_face is None:
            self.status_label.configure(text="Error: Please capture your face first", fg='red')
            return
        
        user_path = os.path.join(self.users_dir, f"{name}.jpg")
        if os.path.exists(user_path):
            self.status_label.configure(text="Error: User already exists", fg='red')
            return
        
        try:
            # Save the cropped face
            cv2.imwrite(user_path, self.captured_face)
            self.status_label.configure(text=f"User '{name}' registered successfully!", fg='green')
            self.name_entry.delete(0, tk.END)
            self.captured_image = None
            self.captured_face = None
            
        except Exception as e:
            self.status_label.configure(text=f"Error: {str(e)}", fg='red')
    
    def verify_face(self):
        """Verify the cropped face against stored user using the trained model"""
        name = self.login_name_entry.get().strip()
        
        if not name:
            self.login_status_label.configure(text="Error: Please enter a name", fg='red')
            return
        
        if self.current_frame is None:
            self.login_status_label.configure(text="Error: No camera feed available", fg='red')
            return
        
        user_path = os.path.join(self.users_dir, f"{name}.jpg")
        if not os.path.exists(user_path):
            self.login_status_label.configure(text="Error: User not found", fg='red')
            return
        
        if self.model is None:
            self.login_status_label.configure(text="Error: Model not loaded", fg='red')
            return
        
        try:
            self.login_status_label.configure(text="Verifying...", fg='blue')
            self.root.update()
            
            # Load stored cropped face
            stored_image = cv2.imread(user_path)
            stored_processed = self.preprocess_image(stored_image)
            
            # Crop and preprocess current frame
            cropped_face = self.crop_face(self.current_frame)
            if cropped_face is None:
                self.login_status_label.configure(text="Error: Failed to detect and crop face", fg='red')
                return
            current_processed = self.preprocess_image(cropped_face)
            
            if stored_processed is None or current_processed is None:
                self.login_status_label.configure(text="Error: Failed to preprocess images", fg='red')
                return
            
            # Perform anti-spoofing and verification
            [verify_pred, spoof_pred] = self.model.predict([current_processed, stored_processed, current_processed])
            
            # Get confidence scores
            spoof_confidence = spoof_pred[0][0] * 100  # Convert to percentage
            verify_confidence = verify_pred[0][0] * 100  # Convert to percentage
            
            # Anti-spoofing check (spoof_pred > 0.5 indicates real face)
            if spoof_pred[0][0] <= 0.5:
                self.login_status_label.configure(
                    text=f"Error: Possible spoofing attempt detected (Confidence: {spoof_confidence:.2f}%)", 
                    fg='red'
                )
                return
            
            # Verification check (verify_pred > 0.5 indicates same person)
            if verify_pred[0][0] > 0.65:
                self.login_status_label.configure(
                    text=f"Welcome {name}! Login successful! (Verification: {verify_confidence:.2f}%, Anti-spoof: {spoof_confidence:.2f}%)", 
                    fg='green'
                )
                messagebox.showinfo(
                    "Success", 
                    f"Welcome {name}! You have been successfully authenticated.\nVerification Confidence: {verify_confidence:.2f}%\nAnti-spoofing Confidence: {spoof_confidence:.2f}%"
                )
            else:
                self.login_status_label.configure(
                    text=f"Error: Face does not match (Verification: {verify_confidence:.2f}%, Anti-spoof: {spoof_confidence:.2f}%)", 
                    fg='red'
                )
                
        except Exception as e:
            self.login_status_label.configure(text=f"Error: {str(e)}", fg='red')
    
    def on_closing(self):
        """Handle window closing"""
        self.stop_camera()
        self.root.destroy()

def main():
    try:
        import cv2
        import tensorflow
    except ImportError as e:
        print("Required libraries not found. Please install:")
        print("pip install opencv-python pillow tensorflow")
        return
    
    root = tk.Tk()
    app = FaceAuthSystem(root)
    root.protocol("WM_DELETE_WINDOW", app.on_closing)
    root.mainloop()

if __name__ == "__main__":
    main()