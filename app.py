from flask import Flask, render_template, request, jsonify, url_for
import os
import pickle
import time
import base64
import uuid
import dlib
import face_recognition
import cv2
import imutils

app = Flask(__name__)

@app.route('/', methods = ['GET','POST'])
def getface():
	if request.method == 'POST':

		# Get the DataURL of the frame posted by canvas.toDataURL and decode the image
		image_url_b64 = next(iter(request.form.to_dict()))
		image_url_b64 = image_url_b64.split(';')[1]
		image_content = image_url_b64.split(',')[1]
		image_decoded_binary = base64.decodebytes(image_content.encode('utf-8'))
		
		# Save extracted decoded image with a unique generated filename
		filename = str(uuid.uuid4().hex)
		with open('uploads/'+ filename +'.jpg', 'wb') as img_file:
			img_file.write(image_decoded_binary)

		# Load the known faces and embeddings
		data = pickle.loads(open("encodings.pickle", "rb").read())

		# Load the input image and convert it from BGR to RGB
		image = cv2.imread('uploads/'+ filename +'.jpg')
		rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

		# Detect the (x, y)-coordinates of the bounding boxes corresponding
		# to each face in the input image.
		boxes = face_recognition.face_locations(rgb, model='hog')

		print('******************************')
		print('no of faces : ', len(boxes))
	
		# If no face is detected or more than 1 faces are detected, then return
		# with corresponding error code
		if len(boxes) == 0: 
			return jsonify(status='error_1')
		if (len(boxes) > 1):
			return jsonify(status='error_2')

		# compute the facial embeddings for the face
		encodings = face_recognition.face_encodings(rgb, boxes)
		
		# encodings is a list of 128 features for every face detected. We will be
		# considering only 1 face so take the first and only list 
		encoding = encodings[0]

		# attempt to match the face in the input image to our known
		# encodings
		matches = face_recognition.compare_faces(data["encodings"], encoding, tolerance=0.4)
		name = "Unknown"

		# check to see if we have found a match
		if True in matches:
			# find the indexes of all matched faces then initialize a
			# dictionary to count the total number of times each face
			# was matched
			matchedIdxs = [i for (i, b) in enumerate(matches) if b]
			counts = {}

			# loop over the matched indexes and maintain a count for
			# each recognized face
			for i in matchedIdxs:
				name = data["names"][i]
				counts[name] = counts.get(name, 0) + 1

			# determine the recognized face with the largest number of
			# votes (note: in the event of an unlikely tie Python will
			# select first entry in the dictionary)
			name = max(counts, key=counts.get)

		if name == 'Unknown':
			return jsonify(status='error_3')
		else:
			return jsonify(status='success', username=name)

	return render_template('landing.html')

if __name__ == '__main__':
	app.run(debug=True)