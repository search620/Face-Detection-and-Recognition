# Face-Detection-and-Recognition
Script for Face Detection and Recognition

This repository contains a Python script that utilizes OpenCV, RetinaFace, and face_recognition libraries to perform sophisticated face detection and recognition tasks. It's designed to efficiently process a large set of images, detect faces, and recognize individuals based on a target image. Here are the key features:

**Face Detection:** Leverages the RetinaFace model for accurate face detection in images.
**Face Recognition:** Utilizes the face_recognition library to identify specific individuals by comparing detected faces against a target image.
**Image Normalization:** Applies histogram equalization to the luminance channel of images to enhance recognition accuracy.
**Visual Feedback:** Draws bounding boxes around detected faces, color-coded to distinguish between recognized and unrecognized faces.
**Concurrent Processing:** Employs concurrent processing techniques to handle large image datasets efficiently.
**Customizable:** Allows users to specify input and output directories, target face images, and toggle the face recognition feature.
This script is ideal for research projects requiring face detection and recognition capabilities.

