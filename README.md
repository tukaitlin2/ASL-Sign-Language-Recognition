# ASL Sign Language Recognition
A real-time ASL alphabet recognition using MediaPipe, OpenCV, and Random Forest.

## Features
- Real-time hand tracking using MediaPipe
- Webcam input using OpenCV
- Recognition of all 26 letters in the ASL alphabet
- Hand landmark normalization for consistent predictions
- Random Forest machine learning classification

## Technologies
- Python
- MediaPipe
- OpenCV
- Pandas
- Scikit-learn
- Joblib

## How It Works
The system uses a webcam to capture the user's hand and MediaPipe's Hand Landmarker to detect 21 landmarks on the hand. Each landmark contains x, y, and z coordinates that are centered on the wrist and scaled to ensure that the data is consistent despite hand sizes and positions. The data is then stored into a CSV file that is used to train a Random Forest classifier. In real-time, the webcam captures the hand position, extracts the landmarks, and passes the data into the trained model to predict the corresponding ASL letter.

## Data Collection
I collected approximately 150 samples for each letter of the ASL alphabet. 
The final dataset contains 3,899 samples across all 26 letters along with 63 landmark features.

## Machine Learning Model
A Random Forest Classifier was traine dusing the collected hand landmark data.
The data was split into:
- 80% training data
- 20% testing data

The model was trained using 100 decision trees.

## Results
The model achieved approximately **99.1%**.
The real-time predictions work successfully through the wecam, however, some letters can produce inconsistent predicitons.

## Future Improvements
- Add prediciton smoothing to reduce inconsistent predictions
- Collect more varied data including different hand positions and orientations
- Improve the recognition of dynamic letters such as J and Z
- Test model with additional users
- Improve real-time prediciton stability

## Project Structure
'''test
ASL-Sign-Language-Recognition/
├── collect_data.py
├── train_model.py
├── predict.py
├── asl_data.csv
├── asl_model.pkl
├── hand_landmarker.task
└── README.md