# Face-photo-sketch-synthesis-and-Recognition
In this project we aim to present a simple Face Photo-Sketch Synthesis and Recognition
system. FSS(Face Sketch Synthesis) provides a way to compare and match the faces present
in two different modalities (i.e face-photos and face-sketches). We can reduce the difference
between photo and sketch significantly and decrease the texture irregularity between them
by converting the photo to a sketch or vice-versa. This results in effective matching between
the two thus simplifying the process of recognition.
This system is modeled using three major components:
i) For a given input face-photo, obtaining an output face-sketch
ii) For a given input face-sketch, obtaining an output face-photo
iii) Recognition of the face-photo or the face-sketch in the database for a given query
face-sketch or face-photo respectively
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
ORL Dataset 
### Commands : 
	python3 Face_recognition.py

