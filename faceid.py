# Import kivy dependencies first
from kivy.app import App
from kivy.uix.boxlayout import BoxLayout
from kivy.core.window import Window

# Import kivy UX components
from kivy.uix.image import Image
from kivy.uix.button import Button
from kivy.uix.label import Label
from kivy.uix.spinner import Spinner
from kivy.uix.screenmanager import ScreenManager, Screen
from kivy.uix.textinput import TextInput

# Import other kivy stuff
from kivy.clock import Clock
from kivy.graphics.texture import Texture
from kivy.logger import Logger

# Import other dependencies
import cv2
import tensorflow as tf
from layers import L1Dist
import os
import numpy as np
import uuid

# Set dark theme background and fixed window size
Window.clearcolor = (0.08, 0.08, 0.1, 1)
Window.size = (500, 750)

class VerificationScreen(Screen):
    def __init__(self, **kwargs):
        super(VerificationScreen, self).__init__(**kwargs)
        # Main layout components 
        layout = BoxLayout(orientation='vertical', padding=20, spacing=20)
        
        # Add a title label
        self.title_label = Label(
            text="[b]Face Verification[/b]", 
            markup=True,
            font_size='28sp', 
            size_hint=(1, 0.1),
            color=(0.9, 0.9, 0.9, 1)
        )
        
        # Load existing profiles for spinner
        profiles = []
        if os.path.exists(os.path.join('application_data', 'verification_profiles')):
            profiles = os.listdir(os.path.join('application_data', 'verification_profiles'))
        if not profiles:
            profiles = ['George_Bush']
            
        self.profile_spinner = Spinner(
            text=profiles[0] if profiles else 'No Profiles',
            values=tuple(profiles),
            size_hint=(1, 0.1),
            font_size='20sp',
            background_color=(0.3, 0.3, 0.4, 1)
        )

        self.web_cam = Image(size_hint=(1, 0.45))
        
        # Style the verify button
        self.button = Button(
            text="VERIFY IDENTITY", 
            on_press=self.verify, 
            size_hint=(1, 0.12),
            font_size='22sp',
            background_normal='',
            background_color=(0.15, 0.5, 0.8, 1),
            bold=True
        )
        
        # Style the verification label
        self.verification_label = Label(
            text="System Ready - Click Verify", 
            size_hint=(1, 0.1),
            font_size='20sp',
            color=(0.6, 0.6, 0.6, 1)
        )
        
        self.nav_button = Button(
            text="Register New User ->", 
            on_press=self.go_to_register, 
            size_hint=(1, 0.08),
            font_size='16sp',
            background_normal='',
            background_color=(0.3, 0.3, 0.3, 1)
        )

        # Add items to layout
        layout.add_widget(self.title_label)
        layout.add_widget(self.profile_spinner)
        layout.add_widget(self.web_cam)
        layout.add_widget(self.button)
        layout.add_widget(self.verification_label)
        layout.add_widget(self.nav_button)
        
        self.add_widget(layout)

    def go_to_register(self, *args):
        self.manager.current = 'register'

    def update_frame(self, frame):
        # Draw a subtle bounding box for guidance
        cv2.rectangle(frame, (1, 1), (248, 248), (0, 255, 0), 2)
        # Flip horizontally and convert image to texture
        buf = cv2.flip(frame, 0).tobytes()
        img_texture = Texture.create(size=(frame.shape[1], frame.shape[0]), colorfmt='bgr')
        img_texture.blit_buffer(buf, colorfmt='bgr', bufferfmt='ubyte')
        self.web_cam.texture = img_texture

    def update_profiles(self, new_profile):
        values = list(self.profile_spinner.values)
        if new_profile not in values:
            values.append(new_profile)
        self.profile_spinner.values = tuple(values)
        self.profile_spinner.text = new_profile

    # Load image from file and conver to 100x100px
    def preprocess(self, file_path):
        # Read in image from file path
        byte_img = tf.io.read_file(file_path)
        # Load in the image 
        img = tf.io.decode_jpeg(byte_img)
        
        # Preprocessing steps - resizing the image to be 100x100x3
        img = tf.image.resize(img, (100,100))
        # Scale image to be between 0 and 1 
        img = img / 255.0
        
        # Return image
        return img

    def verify(self, *args):
        # Specify strict thresholds
        detection_threshold = 0.5
        verification_threshold = 0.5

        # Create directories if they don't exist
        os.makedirs(os.path.join('application_data', 'input_image'), exist_ok=True)
        
        # Get selected profile
        selected_profile = self.profile_spinner.text
        verification_dir = os.path.join('application_data', 'verification_profiles', selected_profile)
        os.makedirs(verification_dir, exist_ok=True)

        # Capture input image from our webcam
        SAVE_PATH = os.path.join('application_data', 'input_image', 'input_image.jpg')
        
        # We need the app's latest clean frame
        app = App.get_running_app()
        if app.latest_frame is not None:
            cv2.imwrite(SAVE_PATH, app.latest_frame)
        else:
            return [], False

        # Preprocess the input image once to save redundant IO and processing
        input_img = self.preprocess(SAVE_PATH)

        # Batch process all validation images efficiently
        val_images_list = os.listdir(verification_dir)
        
        if not val_images_list:
            Logger.warning(f"No images found in {verification_dir}.")
            self.verification_label.text = f'Unverified (Add images to {selected_profile})'
            self.verification_label.color = (1, 0.5, 0, 1) # Orange
            return [], False
            
        val_images = [self.preprocess(os.path.join(verification_dir, file)) for file in val_images_list]

        # Stack into batches for efficient model prediction
        input_images_batch = tf.stack([input_img] * len(val_images))
        val_images_batch = tf.stack(val_images)

        # Make Predictions simultaneously
        results = app.model.predict([input_images_batch, val_images_batch])
        
        # Verification Threshold: Proportion of positive predictions / total positive samples 
        detection = np.sum(np.array(results) > detection_threshold)
        verification = detection / len(val_images_list) 
        verified = verification > verification_threshold

        # Set verification text and color based on the results
        if verified:
            self.verification_label.text = f'VERIFIED (Score: {verification:.2f})'
            self.verification_label.color = (0.2, 0.8, 0.2, 1) # Green
            self.button.background_color = (0.2, 0.8, 0.2, 1) 
        else:
            self.verification_label.text = f'UNVERIFIED (Score: {verification:.2f})'
            self.verification_label.color = (0.9, 0.2, 0.2, 1) # Red
            self.button.background_color = (0.9, 0.2, 0.2, 1)

        # Log out details
        Logger.info(results)
        Logger.info(detection)
        Logger.info(verification)
        Logger.info(verified)
        
        return results, verified

class RegistrationScreen(Screen):
    def __init__(self, **kwargs):
        super(RegistrationScreen, self).__init__(**kwargs)
        layout = BoxLayout(orientation='vertical', padding=20, spacing=20)
        
        self.title_label = Label(
            text="[b]Register New Profile[/b]", 
            markup=True,
            font_size='28sp', 
            size_hint=(1, 0.1),
            color=(0.9, 0.9, 0.9, 1)
        )
        
        self.name_input = TextInput(
            hint_text='Enter Profile Name (e.g. John_Doe)',
            size_hint=(1, 0.1),
            font_size='20sp',
            multiline=False
        )
        
        self.web_cam = Image(size_hint=(1, 0.45))
        
        self.register_button = Button(
            text="START AUTO-CAPTURE", 
            on_press=self.start_registration, 
            size_hint=(1, 0.12),
            font_size='22sp',
            background_normal='',
            background_color=(0.8, 0.4, 0.1, 1),
            bold=True
        )
        
        self.status_label = Label(
            text="Type a name, align face, and Start.", 
            size_hint=(1, 0.1),
            font_size='18sp',
            color=(0.6, 0.6, 0.6, 1)
        )
        
        self.nav_button = Button(
            text="<- Back to Verification", 
            on_press=self.go_to_verify, 
            size_hint=(1, 0.08),
            font_size='16sp',
            background_normal='',
            background_color=(0.3, 0.3, 0.3, 1)
        )

        layout.add_widget(self.title_label)
        layout.add_widget(self.name_input)
        layout.add_widget(self.web_cam)
        layout.add_widget(self.register_button)
        layout.add_widget(self.status_label)
        layout.add_widget(self.nav_button)
        
        self.add_widget(layout)
        
        self.capture_count = 0
        self.capture_target = 10
        self.profile_dir = ""

    def go_to_verify(self, *args):
        self.manager.current = 'verify'

    def update_frame(self, frame):
        # Draw a subtle bounding box for guidance
        cv2.rectangle(frame, (1, 1), (248, 248), (0, 255, 0), 2)
        # Flip horizontally and convert image to texture
        buf = cv2.flip(frame, 0).tobytes()
        img_texture = Texture.create(size=(frame.shape[1], frame.shape[0]), colorfmt='bgr')
        img_texture.blit_buffer(buf, colorfmt='bgr', bufferfmt='ubyte')
        self.web_cam.texture = img_texture

    def start_registration(self, *args):
        profile_name = self.name_input.text.strip().replace(" ", "_")
        if not profile_name:
            self.status_label.text = "Please enter a valid name!"
            self.status_label.color = (1, 0, 0, 1)
            return
            
        self.profile_dir = os.path.join('application_data', 'verification_profiles', profile_name)
        os.makedirs(self.profile_dir, exist_ok=True)
        
        self.capture_count = 0
        self.register_button.disabled = True
        self.name_input.disabled = True
        self.status_label.color = (0.2, 0.8, 0.2, 1)
        
        # Schedule the capture loop
        Clock.schedule_interval(self.capture_image, 0.3)
        
    def capture_image(self, dt):
        app = App.get_running_app()
        if app.latest_frame is not None:
            # Save the frame
            filename = os.path.join(self.profile_dir, f'{uuid.uuid1()}.jpg')
            cv2.imwrite(filename, app.latest_frame)
            self.capture_count += 1
            self.status_label.text = f"Capturing: {self.capture_count}/{self.capture_target} images..."
            
            if self.capture_count >= self.capture_target:
                # Stop scheduling
                Clock.unschedule(self.capture_image)
                self.status_label.text = "Registration Complete! Auto-added to Verification."
                self.register_button.disabled = False
                self.name_input.disabled = False
                
                # Update spinner in verify screen
                verify_screen = self.manager.get_screen('verify')
                verify_screen.update_profiles(self.name_input.text.strip().replace(" ", "_"))
        
class CamApp(App):
    def build(self):
        # Load tensorflow/keras model globally so it's accessible
        self.model = tf.keras.models.load_model('siamesemodelv3.h5', custom_objects={'L1Dist':L1Dist})
        
        # Setup Screen Manager
        self.sm = ScreenManager()
        self.verify_screen = VerificationScreen(name='verify')
        self.register_screen = RegistrationScreen(name='register')
        
        self.sm.add_widget(self.verify_screen)
        self.sm.add_widget(self.register_screen)
        
        # Setup video capture device globally
        self.capture = cv2.VideoCapture(0)
        self.latest_frame = None
        Clock.schedule_interval(self.update, 1.0/33.0)
        
        return self.sm

    # Run continuously to get webcam feed
    def update(self, *args):
        ret, frame = self.capture.read()
        if ret:
            # Crop frame to standard 250x250
            frame = frame[120:120+250, 200:200+250, :]
            self.latest_frame = frame.copy()
            
            # Send frame to the currently active screen
            if self.sm.current == 'verify':
                self.verify_screen.update_frame(frame.copy())
            elif self.sm.current == 'register':
                self.register_screen.update_frame(frame.copy())

if __name__ == '__main__':
    CamApp().run()
