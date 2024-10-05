# GestureControl
This project implements a hand gesture-based volume control system using a webcam. The system utilizes MediaPipe for hand landmark detection and pycaw to control the system's volume based on the distance and movement of specific hand landmarks (thumb and index finger).
**Features**
Real-time hand gesture recognition using the webcam.
Dynamic volume control based on the distance between the thumb and index finger.
Fine-tuned volume adjustment using vertical hand movements.
Customizable sensitivity and scaling for volume changes

**Prerequisites**
Before running the project, ensure you have installed the following dependencies:

Python 3.x
OpenCV (cv2)
MediaPipe (mediapipe)
NumPy (numpy)
Pycaw (pycaw)
Comtypes (comtypes)
You can install the dependencies using the following command:
*pip install opencv-python mediapipe numpy pycaw comtypes*

**Usage**
Clone the repository:

*git clone <repository_url>*
*cd <repository_folder>*


Run the Python script:

*python hand_volume_control.py*
The webcam will open, and you can control the volume by:

Pinching your thumb and index finger together to change the volume.
Moving your hand upwards to increase the volume or downwards to decrease it.
Press 'q' to exit the program.

**Key Components**
MediaPipe Hands: Used for detecting hand landmarks such as thumb and index finger.
Pycaw: Interface with system audio to control the volume based on detected gestures.
OpenCV: Captures and processes video frames from the webcam.
Code Breakdown
get_distance(): Calculates the distance between two hand landmarks.
control_volume(): Adjusts the system volume based on hand position and movement.
main(): Captures webcam feed, processes hand gestures, and triggers volume control.
Customization
Sensitivity: Adjust the value of sensitivity in the main() function to change the hand movement sensitivity (default is 0.02).
Scaling Factor: Modify scaling_factor to fine-tune the volume adjustment rate (default is 0.2).


**Error Handling**
If any of the required libraries are missing, the script will output an appropriate error message, prompting you to install the missing dependencies.
