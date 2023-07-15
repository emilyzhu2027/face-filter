"""
Emily Zhu --
This code uses OpenCV library to detect if and where there are faces in frame and places
an image over the top of the face.
"""
import cv2

# Import the file to detect frontal faces
cascadeClassifier = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

# Start capturing livestream video
vidCapture = cv2.VideoCapture(0)

# Import "filter" image that will overlay any faces in frame
filterImg = cv2.imread("sample_img.jpeg", cv2.IMREAD_UNCHANGED)
# Get the dimensions of the image to scale it later
hOriginal, wOriginal, cOriginal = filterImg.shape

while True:
    # Read the frame of the livestream video
    _, frame = vidCapture.read()
    # Flip frame on y-axis
    frame = cv2.flip(frame, 1)
    fheight, fwidth, fc = frame.shape
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect any faces
    faces = cascadeClassifier.detectMultiScale(gray, 1.1, 4)

    # Loop through each face detected
    for (x, y, w, h) in faces:
        # Rescaling the filter image
        scaleHoverW = hOriginal/wOriginal
        resizedFilterH = int(scaleHoverW * w)
        resizedFilterW = int(w)

        # Get bounds of the filter image on the frame
        topImg = int(y)-resizedFilterH
        bottomImg = int(y)
        leftImg = int(x)
        rightImg = int(x)+resizedFilterW

        # Check if the filter image is within the bounds of the frame
        if (topImg > 1 and bottomImg < int(fheight - 1) and leftImg > 1 and rightImg < int(fwidth - 1)):
            # If within bounds of the frame, resize the filter image
            filterResize = cv2.resize(filterImg, (resizedFilterW, resizedFilterH), interpolation = cv2.INTER_AREA)
            # And then replace the pixels of the frame with the filter image
            frame[topImg:bottomImg, leftImg:rightImg] = filterResize

    # Display the frame of the video
    cv2.imshow('img', frame)

    # Stop livestream video if esc key is pressed
    k = cv2.waitKey(30) & 0xff
    if k==27:
        break

# Release the Videocapture object
vidCapture.release()
