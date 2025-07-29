# Face Verification System

## Overview

This face verification system authenticates users by comparing facial features using deep learning models. The system provides both a GUI application and REST API for face recognition with high accuracy across various conditions.

## Features

- Real-time face verification
- User-friendly GUI interface
- RESTful API for integration
- High accuracy face recognition

## Technologies Used

![Python](https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54)
![TensorFlow](https://img.shields.io/badge/TensorFlow-%23FF6F00.svg?style=for-the-badge&logo=TensorFlow&logoColor=white)
![Keras](https://img.shields.io/badge/Keras-%23D00000.svg?style=for-the-badge&logo=Keras&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-%23F7931E.svg?style=for-the-badge&logo=scikit-learn&logoColor=white)
![FastAPI](https://img.shields.io/badge/FastAPI-005571?style=for-the-badge&logo=fastapi)
![Tkinter](https://img.shields.io/badge/Tkinter-3776AB?style=for-the-badge&logo=python&logoColor=white)

## Requirements

```
tensorflow>=2.10.0
keras>=2.10.0
scikit-learn>=1.1.0
fastapi>=0.95.0
tkinter>=8.6
opencv-python>=4.7.0
numpy>=1.21.0
pillow>=9.0.0
uvicorn>=0.20.0
```

## Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/face-verification-system.git
cd face-verification-system

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## Usage

### GUI Application
```bash
python GUI.py
```

### API Server
```bash
uvicorn server:app --reload
```


## GUI Overview

The application features a user-friendly graphical interface built with Tkinter that allows users to:

- **Image Upload**: Browse and select images for face verification
- **Real-time Camera**: Capture photos directly from webcam for instant verification
- **Verification Results**: Display match confidence scores and authentication status
- **Database Management**: Add new faces to the recognition database

The interface is designed with intuitive controls and clear visual feedback to ensure smooth user experience.

## API Endpoints

- `GET /api/metrics` - Get system performance metrics
- `POST /api/predictions` - Perform face verification prediction

## Model Performance

![ROC Curve](ROC_curve.png)

*The ROC curve demonstrates the system's performance with an AUC score indicating excellent classification accuracy between genuine and imposter face pairs.*

## Contact

Tran Buu Duc Tri - yductri02lt2@gmail.com

Project Link: [https://github.com/yourusername/face-verification-system](https://github.com/yourusername/face-verification-system)