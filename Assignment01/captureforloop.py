import cv2
import time
import numpy as np

#use 0 for computer camera and 1 for phone camera
cap = cv2.VideoCapture(0)

while True:
    # Start measuring time 
    start_time = time.time()

    # Capture frame-by-frame
    ret, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect brightest spot using double for-loop
    brightest_value = -1
    brightest_loc = (0, 0)
    for y in range(gray.shape[0]):
        for x in range(gray.shape[1]):
			#check if the current pixel is higher than the current brightest_value
            #if yes, store it as the brightest_value
            if gray[y, x] > brightest_value:
                brightest_value = gray[y, x]
                brightest_loc = (x, y)

    # Draw blue circle at the brightest spot
    cv2.circle(frame, brightest_loc, 10, (255, 0, 0), 2)

    # Detect reddest spot using double for-loop
    # OpenCv stores color in BGR format so the first channel is blue, second channel is the green and the third chanel is the red
    blue_channel = frame[:, :, 0]
    green_channel = frame[:, :, 1]
    red_channel = frame[:, :, 2]

    reddest_value = -np.inf
    reddest_loc = (0, 0)
    for y in range(frame.shape[0]):
        for x in range(frame.shape[1]):
            # Calculate the intensity of the red value 
            red_intensity = int(red_channel[y, x]) - int((green_channel[y, x] + blue_channel[y, x]) // 2)
            # if the red intensity is higher than the current reddest value store it as the reddest value
            if red_intensity > reddest_value:
                reddest_value = red_intensity
                reddest_loc = (x, y)

    #Draw red circle at the reddest spot
    cv2.circle(frame, reddest_loc, 10, (0, 0, 255), 2)

    #calculate FPS
    end_time = time.time()
    frame_time = end_time - start_time
    fps = 1 / (end_time - start_time)

    
    cv2.putText(frame, f"FPS: {fps:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

    print(f"Frame processing time: {frame_time:.4f} seconds | FPS: {fps:.2f}")

    cv2.imshow('frame', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
