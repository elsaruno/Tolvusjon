import cv2
import time
import numpy as np
from datetime import datetime

#use 0 for computer camera and 1 for phone camera
cap = cv2.VideoCapture(0)

while(True):
	# Start measuring time
	start_time = time.time()

	# Capture frame-by-frame
	ret, frame = cap.read()
	gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

	# Locate brightest spot 
	min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(gray)

	# Draw blue circle at brightest spot 
	cv2.circle(frame, max_loc, 10, (255, 0, 0), 2)
	
	# Locate the reddest spot, highest intensity in the red channel minus intensity in green and blue
	# OpenCv stores color in BGR format so the first channel is blue, second channel is the green and the third chanel is the red
	blue_channel, green_channel, red_channel = cv2.split(frame)
	blue_channel = blue_channel.astype(int)
	green_channel = green_channel.astype(int)
	red_channel = red_channel.astype(int)

	red_difference = red_channel - ((green_channel + blue_channel) // 2)
	reddest_spot = np.unravel_index(np.argmax(red_difference), red_difference.shape)

	#Draw red circle for the reddest spot
	cv2.circle(frame, (reddest_spot[1], reddest_spot[0]), 10, (0, 0, 255), 2)

	#calculate FPS
	end_time = time.time()
	frame_time = end_time - start_time
	fps = 1 / (end_time - start_time)
	
	timestamp = datetime.now().strftime("%H:%M:%S")

	cv2.putText(frame, f"FPS: {fps:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
	cv2.putText(frame, f"Timestamp: {timestamp}", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

	print(f"Frame processing time: {frame_time:.4f} seconds | FPS: {fps:.2f}")

	cv2.imshow('frame', frame)
	if cv2.waitKey(1) & 0xFF == ord('q'):
		break

cap.release()
cv2.destroyAllWindows()