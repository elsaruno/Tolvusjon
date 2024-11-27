from sklearn.linear_model import RANSACRegressor
import cv2
import numpy as np


# Open video capture
cap = cv2.VideoCapture(0)  # 0 for default camera

# Set parameters for Canny (found parameters that make sense using canny_test.py)
canny_threshold1 = 50
canny_threshold2 = 150

while True:
	ret, frame = cap.read()
	if not ret:
		break
	
	gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
	
	# Apply Canny edge detection
	edges = cv2.Canny(gray, canny_threshold1, canny_threshold2)
	
	# Get coordinates of edge pixels (those are the pixels that have intensity higher than 0)
	y_indices, x_indices = np.where(edges > 0)
	points = np.column_stack((x_indices, y_indices))
	
	if len(points) > 2:  
		# Use sklearn's RANSAC
		ransac = RANSACRegressor()
		# fit x and y indices using RANSAC algorithm
		ransac.fit(points[:, 0].reshape(-1, 1), points[:, 1])  # Fit x to y

		# Get line parameters
		# estimator_ = Best fitted model (copy of the estimator object).
		# coef_ gives an array of weights estimated by linear regression, the first value is the slope of the line
		# intercept_ gives the intercept or bias term 
		slope = ransac.estimator_.coef_[0]
		intercept = ransac.estimator_.intercept_
		
		# Define two points for the line
		# x1 is set to 0 to start at the left edge of the image
		# x2 is set to the widh of the image to be at the right edge of the image 
		x1, x2 = 0, frame.shape[1]
		# use equation for straight line y = slope * x + intercept to determine y1 and y2 (x1 = 0 so y1 is just y = intercept)
		y1, y2 = int(intercept), int(slope * x2 + intercept)
		
		# Draw the line from x1,y1 to x2,y2 and have it green
		cv2.line(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
	
	
	cv2.imshow("Line Detection", frame)
	
	if cv2.waitKey(1) & 0xFF == ord('q'):
		break

cap.release()
cv2.destroyAllWindows()

