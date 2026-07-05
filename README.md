# Sign_Language_Detection-Task-4
A machine learning project that detects sign language gestures from uploaded images and webcam using YOLOv8.

# Project Overview
This project is a Sign Language Detection System developed using Python, YOLOv8, OpenCV, and Streamlit. The system detects predefined sign language gestures from uploaded images and live webcam video.

The application works only between 6:00 PM and 10:00 PM as per the project requirement.

# Features
•	Detects sign language from uploaded images.
•	Supports multiple image upload.
•	Detects sign language using a live webcam.
•	Displays the detected sign with confidence score.
•	User-friendly graphical interface using Streamlit.
•	Application works only between 6 PM and 10 PM.

# Supported Signs
The model detects the following signs:
•	Hello
•	Yes
•	No
•	Thank You
•	I Love You

# Technologies Used
•	Python
•	YOLOv8
•	OpenCV
•	Streamlit
•	NumPy
•	Pillow

# Project Structure
•	Sign_Language_task
•	dataset_split
o	images
o	labels
•	best.py
•	data.yaml
•	train.py
•	predict.py
•	sign_language.py

# Dataset
The model is trained using a 5-class Sign Language Detection Dataset.
Classes:
•	Hello
•	Yes
•	No
•	Thank You
•	I Love You

# How the Project Works
# Step 1
Train the YOLOv8 model using the training dataset.

# Step 2
The trained model is saved as:
best.pt

# Step 3
Run the Streamlit application.

# Step 4
Choose one of the following options:
•	Upload Images
•	Webcam

# Step 5
The model predicts the sign language gesture.

# Step 6
The detected sign and confidence score are displayed.

# How to Run the Project
1. Clone the Repository
git clone <repository_link>

2. Open the Project Folder
cd Sign_Language_Task

3. Install Required Libraries
pip install -r requirements.txt

4. Run the Application
streamlit run sign_language.py

# Output
The application displays:
•	Uploaded image or webcam frame
•	Detected sign
•	Confidence score

# Time Restriction
The application is available only between:
6:00 PM to 10:00 PM
Outside this time, the application displays:
Application works only between 6:00 PM and 10:00 PM.


