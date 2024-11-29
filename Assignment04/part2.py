import cv2
import numpy as np

# find the intersection of lines using cross product 
def find_intersections(lines):
    #function to convert two points into a line homogeneous cordinates
    def to_homogeneous(point1, point2):
        return np.cross([point1[0], point1[1], 1], [point2[0], point2[1], 1])
	#function to convert point from homogeneous to cartesian coordinate
    def from_homogeneous(h_point):
        if h_point[2] == 0:  # Check if point is at infinity
            return None
        x = h_point[0] / h_point[2]
        y = h_point[1] / h_point[2]
        return int(x), int(y)

    intersections = []
    for i in range(len(lines)):
        for j in range(i + 1, len(lines)):
            # Convert lines to homogeneous coordinates
            line1 = to_homogeneous((lines[i][0], lines[i][1]), (lines[i][2], lines[i][3]))
            line2 = to_homogeneous((lines[j][0], lines[j][1]), (lines[j][2], lines[j][3]))
            # Compute the intersection point
            intersection = np.cross(line1, line2)
            cartesian_point = from_homogeneous(intersection)
            if cartesian_point is not None:
                intersections.append(cartesian_point)
    return intersections

def extend_line(x1, y1, x2, y2, ratio):
    dx, dy = x2 - x1, y2 - y1
    x1_ext = int(x1 - ratio * dx)
    y1_ext = int(y1 - ratio * dy)
    x2_ext = int(x2 + ratio * dx)
    y2_ext = int(y2 + ratio * dy)
    return x1_ext, y1_ext, x2_ext, y2_ext

# Function to calculate the length of a line
def line_length(line):
    x1, y1, x2, y2 = line
    return np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)

# Function to order points for perspective transform
def order_points(points):
    points = np.array(points)
    rect = np.zeros((4, 2), dtype="float32")
    s = points.sum(axis=1)
    diff = np.diff(points, axis=1)

    rect[0] = points[np.argmin(s)]  # Top-left
    rect[1] = points[np.argmin(diff)]  # Top-right
    rect[2] = points[np.argmax(s)]  # Bottom-right
    rect[3] = points[np.argmax(diff)]  # Bottom-left
    return rect

# Function to process the frame
def process_frame(frame):
    frame_height, frame_width = frame.shape[:2]

    # Convert the frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Apply Canny Edge Detection
    edges = cv2.Canny(gray, 50, 150)

    # Use Hough Transform to detect lines
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=80, minLineLength=150, maxLineGap=50)

    intersections = []
    warped = None
    if lines is not None:
        # Calculate line lengths and sort by length
        lines = [line[0] for line in lines]  # Extract line endpoints
        lines = sorted(lines, key=line_length, reverse=True)[:4]  # Select top 4 longest lines 

        # Extend lines by a certain ratio to make intersection easier (lines tend to "break")
        extended_lines = []
        for x1, y1, x2, y2 in lines:
            extended_line = extend_line(x1, y1, x2, y2, 3)
            extended_lines.append(extended_line)

        intersections = find_intersections(extended_lines)
                
        if len(intersections) >= 4: #ensure we have intersection for each corner
            # Order the intersection points
            rect = order_points(intersections[:4])

            # Define destination points for the new perspective
            dst = np.array([
                [0, 0],
                [frame_width - 1, 0],
                [frame_width - 1, frame_height - 1],
                [0, frame_height - 1]
            ], dtype="float32")

            # Compute the perspective transform matrix
            M = cv2.getPerspectiveTransform(rect, dst)

            # Perform the perspective warp
            warped = cv2.warpPerspective(frame, M, (frame_width, frame_height))

        # Draw the extended lines
        for x1, y1, x2, y2 in extended_lines:
            cv2.line(frame, (x1, y1), (x2, y2), (0, 255, 0), 5)

        # Draw the intersection points
        for pt in intersections:
            if 0 <= pt[0] < frame_width and 0 <= pt[1] < frame_height:  # Ensure within frame
                cv2.circle(frame, pt, 5, (0, 0, 255), -1)

    return frame, edges, intersections, warped

# Initialize video capture
cap = cv2.VideoCapture(1) 

while True:
    ret, frame = cap.read()

    # Process the frame to detect lines, intersections, and perform perspective transform
    processed_frame, edges, intersections, warped = process_frame(frame)

    # Display the original frame with lines and intersections
    cv2.imshow('Frame with Lines and Intersections', processed_frame)

    #we can view the edge image to see if the results from the Canny detection is good enough
    cv2.imshow('Edge Detection', edges)

    # Display the warped image if it is available 
    if warped is not None:
        cv2.imshow('Warped image', warped)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
