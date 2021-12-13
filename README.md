# Face-photo-sketch-synthesis-and-Recognition
Conversion of face photo to sketch, face sketch to photo and recognition in both modalities

## Face sketch synthesis 
### Files required 
	Photo-Sketch.py 
	input.jpg 
### Commands: <br>
	python3 Photo-Sketch.py <br>
	(give path to input images in Photo-Sketch.py file in ‘path=’/images’’) <br>

## Face photo synthesis
Files required <br>
haarcascade_frontalface_default.xml <br>
training_output_cropped.npy <br>
training_input_cropped.npy <br>
detect_face.py <br>
fetch_data.py <br>
Photo-Sketch.py <br>
train_model.py <br>
Prediction.py <br>
Commands : <br>
	python3 detect_face.py <br>
	python3 fetch_data.py <br>
	python3 Photo-Sketch.py <br>
	python3 train_model.py <br>
	python3 Prediction.py path_to_sketch.jpg <br>

## Facial recognition
Requirements : <br>
	Orl dataset <br>
Commands : <br>
	python3 Face_recognition.py <br>

