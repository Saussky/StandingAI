# AI-Powered Standing Desk Tracker

## Introduction
This project aims to promote a healthier working environment by tracking the amount of time spent standing at a desk. It employs a basic AI image recognition system using MobileNetV2 as the base model, enhanced with a custom-trained layer for distinguishing between sitting and standing postures.

## Setup
### Prerequisites
- Python 3.x
- Webcam connected to your computer

### Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/your-repository.git

2. Install the required dependencies:
    ```bash
    pip install -r requirements.txt

# Custom Training Data
To train the model with your data:

1. Capture images of yourself in sitting and standing positions.
2. Place an equal number of images in the **images/sitting** and **images/standing** directories.
3. Train the model by running the provided training script.

# Usage

After setting up the project and training the model:

1. Start the application. It will capture an image from the webcam every minute.
2. The AI will classify each image as sitting or standing.
3. The total standing minutes are logged into a CSV file for analysis.

# Data Sorting

*Images with lower confidence scores are stored in the to_sort folder.
*Manually classify these images and move them to the appropriate folders to enhance the model's accuracy.

# Note

The project is currently under development, and features for processing and displaying data are forthcoming.