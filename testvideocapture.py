import cv2
cap = cv2.VideoCapture(0)

while True:

    # Read the frame
    _, img = cap.read()
    print(img)
    # Stop if end of video file
    if _ == False:
        break
    # Stop if escape key is pressed
    key = cv2.waitKey(25) & 0xff
    if key == 27:
        break
# Release the VideoCapture object
cap.release()
