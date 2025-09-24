import speech_recognition as sr
import threading
import time
import subprocess
import sys
import os

class AudioCommandHandler:
    def __init__(self):
        self.recognizer = sr.Recognizer()
        self.is_listening = False
        self.commands = {
            'start tracking': self.start_tracking,
            'stop tracking': self.stop_tracking,
            'calibrate cameras': self.calibrate_cameras,
            'detect markers': self.detect_markers,
            'save data': self.save_data,
            'quit': self.quit_program
        }
    
    def start_listening(self):
        """Start listening for audio commands in a separate thread"""
        self.is_listening = True
        thread = threading.Thread(target=self._listen_loop)
        thread.daemon = True
        thread.start()
        print("Audio commands enabled. Say 'start tracking', 'stop tracking', 'calibrate cameras', 'detect markers', 'save data', or 'quit'")
    
    def _listen_loop(self):
        """Main listening loop"""
        with sr.Microphone() as source:
            # Adjust for ambient noise
            self.recognizer.adjust_for_ambient_noise(source, duration=1)
            
            while self.is_listening:
                try:
                    print("Listening...")
                    audio = self.recognizer.listen(source, timeout=1, phrase_time_limit=5)
                    command = self.recognizer.recognize_google(audio).lower()
                    print(f"Command received: {command}")
                    self.process_command(command)
                except sr.WaitTimeoutError:
                    continue
                except sr.UnknownValueError:
                    continue
                except sr.RequestError as e:
                    print(f"Speech recognition error: {e}")
                    time.sleep(1)
    
    def process_command(self, command):
        """Process the recognized command"""
        for cmd, func in self.commands.items():
            if cmd in command:
                print(f"Executing: {cmd}")
                func()
                return
        print(f"Unknown command: {command}")
    
    def start_tracking(self):
        """Start ArUco tracking"""
        print("Starting ArUco tracking...")
        # Add your tracking start logic here
        # subprocess.run([sys.executable, "tracking/aruco_track.py"])
    
    def stop_tracking(self):
        """Stop ArUco tracking"""
        print("Stopping ArUco tracking...")
        # Add your tracking stop logic here
    
    def calibrate_cameras(self):
        """Start camera calibration"""
        print("Starting camera calibration...")
        # Add your calibration logic here
        # subprocess.run([sys.executable, "CameraCalibration/array_calibration.py"])
    
    def detect_markers(self):
        """Run marker detection"""
        print("Running marker detection...")
        # Add your detection logic here
        # subprocess.run([sys.executable, "aruco_detection/aruco_datagen.py"])
    
    def save_data(self):
        """Save current data"""
        print("Saving data...")
        # Add your data saving logic here
    
    def quit_program(self):
        """Quit the program"""
        print("Quitting program...")
        self.is_listening = False
        sys.exit(0)
    
    def stop_listening(self):
        """Stop listening for commands"""
        self.is_listening = False

# Example usage
if __name__ == "__main__":
    handler = AudioCommandHandler()
    try:
        handler.start_listening()
        # Keep the main thread alive
        while handler.is_listening:
            time.sleep(0.1)
    except KeyboardInterrupt:
        print("Stopping audio commands...")
        handler.stop_listening() 