import numpy as np
import cv2 as cv
import argparse
import csv
import time
def process_frame(frame, color):
    img = frame
    
    assert img is not None, "file could not be read, check with os.path.exists()"
    hsv = cv.cvtColor(img, cv.COLOR_BGR2HSV)
    
    if color == 'red':
        lower_bound1 = np.array([0,100,100])
        upper_bound1 = np.array([8,255,255])

        lower_bound2 = np.array([170,50,50])
        upper_bound2 = np.array([179,255,255])
    elif color == 'yellow':
        #need to change the hsv values for yellow range
        lower_bound1 = np.array([0,100,100])
        upper_bound1 = np.array([9,255,255])

        lower_bound2 = np.array([170,50,50])
        upper_bound2 = np.array([179,255,255])
    else:
        assert "marker color is not supported"

    lower_mask = cv.inRange(hsv, lower_bound1, upper_bound1)
    upper_mask = cv.inRange(hsv, lower_bound2, upper_bound2)
    full_mask = lower_mask + upper_mask

    result=img.copy()
    result = cv.bitwise_and(result, result, mask=full_mask)

    gray=cv.cvtColor(result, cv.COLOR_BGR2GRAY)
    blur = cv.GaussianBlur(gray, (5, 5), cv.BORDER_DEFAULT)
    ret, thresh = cv.threshold(blur, 50, 255, cv.THRESH_BINARY_INV)    

    contours,hierarchy = cv.findContours(thresh, cv.RETR_CCOMP, 1)

    top_margin = 0.22
    bottom_margin = 0.25
    left_margin = 0.3
    right_margin = 0.3
    height, width = img.shape[:2]

    counter=0

    coordinates = np.zeros(shape=(1,2))
    for i in contours:
        M = cv.moments(i)
        if M['m00']>200:
            cx=int(M['m10']/M['m00'])
            cy=int(M['m01']/M['m00'])
            if cx>width*left_margin and cx<width*(1-right_margin) and cy>height*bottom_margin and cy<height*(1-top_margin):
                counter=counter+1
                cv.circle(img, (cx,cy), 7, (0, 255, 255), -1)
                cv.imshow('img',img)
                cv.waitKey(100)
                #print(f"x:{cx} y:{cy}")
                #print(M['m00'])
                coordinates=np.append(coordinates, [[cx,cy]], axis = 0)
                #print(coordinates)
    #print(counter)

    if counter>1:
        return(coordinates[2:coordinates.shape[0],:])
    else:
        return []


parser = argparse.ArgumentParser(
        prog='Marker Detection',
        description='Save the marker coordinates in each fame into CSV',
    )

parser.add_argument('-i', '--input', required=True)
parser.add_argument('-o', '--output', required=False)
parser.add_argument('-c', '--markercolor', required=True) #yellow or red

args = parser.parse_args()

cap=cv.VideoCapture(args.input)
# with open(args.output, 'w', newline='') as csvfile:
# writer=csv.writer(csvfile)
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    
    res = process_frame(frame, args.markercolor)
    # if res!= []:
        # writer.writerow(res.flatten())

    # if cv.waitKey(1) & 0xFF == ord('q'):
        # break
    time.sleep(3)

