# Pascal VOC Pseudo-Labeling with Tensorflow Lite

## Description

This project performs object detection using TensorFlow Lite models. It includes functionality to detect objects in images, generate XML annotations in Pascal VOC format, and visualize bounding boxes on images. **But you need a Tensorflow Lite model** usualy formated .tflite and **labelmap** for this pseudo-labeling to work. this repository provided with a sample model in case you need to know how this project work.

## Requirements

- Python 3.x
- TensorFlow
- OpenCV
- NumPy

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/pukiskun/pseudo-labeling.git
   ```
2. Navigate to the project directory:
   ```bash
   cd pseudo-labeling
   ```
3. Install requirements:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

1. Run the script:
   ```bash
   python main.py
   ```
2. Update the paths in main.py to point to your model, images, and labels
3. Labeling done before you blink.
