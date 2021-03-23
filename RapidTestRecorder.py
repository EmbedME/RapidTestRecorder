# 
# This python script demonstrates the use of Obeltus, a slow running
# conveyor belt, and image analysis for rapid test results analysis.
#
# Obeltus: https://www.fischl.de/obeltus/
#
# Copyright (c) 2021 Thomas Fischl, https://www.fischl.de
# 
# This program is free software: you can redistribute it and/or modify  
# it under the terms of the GNU General Public License as published by  
# the Free Software Foundation, version 3.
#
# This program is distributed in the hope that it will be useful, but 
# WITHOUT ANY WARRANTY; without even the implied warranty of 
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU 
# General Public License for more details.
#
# You should have received a copy of the GNU General Public License 
# along with this program. If not, see <http://www.gnu.org/licenses/>.

import cv2
from pyzbar import pyzbar
import numpy as np

# y coordinate where to start with barcode detection
sbound = 150

# maximum size of list where we store the detected codes
ldetected_size = 6

# prepare capture device and get size of it
cap = cv2.VideoCapture(2)
width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# prepare list for detected codes
ldetected = []

# prepare frame where we show the detected tests
detected = np.zeros((120*ldetected_size,width, 3), np.uint8)

while(True):

    # capture frame from camera
    ret, frame = cap.read()

    # get our region of interest
    roi = frame[sbound:height, 0:width].copy()

    # detect barcodes in it
    barcodes = pyzbar.decode(roi)

    # walk through all detected codes
    for barcode in barcodes:

        # extract the bounding box location of the barcode
        (x, y, w, h) = barcode.rect
        y = y + sbound

        # get data and type of barcode
        barcodeData = barcode.data.decode("utf-8")
        barcodeType = barcode.type

        # check if we have already processed this barcode
        if barcodeData not in ldetected:

            # print the barcode type and data to the terminal
            print("Found {} code: {}".format(barcodeType, barcodeData))

            # add barcode to our list of processed codes
            # limit size of this list
            if len(ldetected) >= ldetected_size:
                ldetected.pop(0)
            ldetected.append(barcodeData)

            # cut out the interesting area from frame
            section = frame[y-40:y+2*40, 0:width].copy()

            # and add it to the detection frame to show the results
            detected = cv2.vconcat([detected[120:120*10, 0:width], section])

            # write it to file system
            filename = "{}.jpg".format(barcodeData)
            cv2.imwrite(filename, section) 

        # draw rectangle around barcode
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)

        # draw barcode data
        text = "{}".format(barcodeData)
        cv2.putText(frame, text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
            

    # draw line to show where detection starts
    cv2.line(frame, (0,sbound),(width,sbound), (255,0,0), 2)

    # Display the resulting frames
    cv2.imshow('Live', frame)
    cv2.imshow('Detected', detected)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# finished, release the capture
cap.release()
cv2.destroyAllWindows()

