# Face-photo-sketch-synthesis-and-Recognition
Conversion of face photo to sketch, face sketch to photo and recognition in both modalities

## Face sketch synthesis 
### Files required 
Photo-Sketch.py 
input.jpg 
### Commands: 
	python3 Photo-Sketch.py 
(give path to input images in Photo-Sketch.py file in ‘path=’/images’’) 

## Face photo synthesis
### Files required 
haarcascade_frontalface_default.xml <br>
training_output_cropped.npy <br>
training_input_cropped.npy <br>
detect_face.py <br>
fetch_data.py <br>
Photo-Sketch.py <br>
train_model.py <br>
Prediction.py <br>
CelebA Dataset 
### Commands : 
	python3 detect_face.py
	python3 fetch_data.py 
	python3 Photo-Sketch.py 
	python3 train_model.py 
	python3 Prediction.py path_to_sketch.jpg 

## Facial recognition
### Requirements : 
ORL dataset 
### Commands : 
	python3 Face_recognition.py

