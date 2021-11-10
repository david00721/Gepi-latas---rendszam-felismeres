import cv2 
import numpy as np 
import imutils
import pytesseract
from matplotlib import pyplot as plt
from difflib import SequenceMatcher
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

def similar(a, b):
    if b is not None:
        return SequenceMatcher(None, a, b).ratio()
    else:
        return 0.0

def get_knownplate(inputpath):
    inputpath = inputpath.replace("rendszam kepek\\\\", "")
    inputpath = inputpath.replace(".jpg", "")
    return inputpath

def order_points(pts):
	rect = np.zeros((4, 2), dtype = "float32")
	s = pts.sum(axis = 1)
	rect[0] = pts[np.argmin(s)]
	rect[2] = pts[np.argmax(s)]
	diff = np.diff(pts, axis = 1)
	rect[1] = pts[np.argmin(diff)]
	rect[3] = pts[np.argmax(diff)]
	return rect

def four_point_transform(image, pts):
	rect = order_points(pts)
	(tl, tr, br, bl) = rect
	widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
	widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
	maxWidth = max(int(widthA), int(widthB))
	heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
	heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
	maxHeight = max(int(heightA), int(heightB))
	dst = np.array([[0, 0],[maxWidth - 1, 0],[maxWidth - 1, maxHeight - 1],[0, maxHeight - 1]], dtype = "float32")
	M = cv2.getPerspectiveTransform(rect, dst)
	warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight))
	return warped

#----------rendszám kinyerő függvény-----------

def numberplate_to_text(img):    
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    #cv2.imshow("Szurkearnyalatos kep", gray)
    #cv2.waitKey(0)
    bfilter = cv2.bilateralFilter(gray, 11, 17, 17)
    edged = cv2.Canny(bfilter, 100, 300) #eddigi legjobb
    #cv2.imshow("Canny detektorral előállított kép", edged)
    #cv2.waitKey(0)

#---------------kontúrkeresés-----------------
    
    keypoints = cv2.findContours(edged.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE) 
    contours = imutils.grab_contours(keypoints)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)[:10] #10 a legjobb

#----------------rendszám megkeresése-----------------------
    location = None
    for contour in contours: 
        approx = cv2.approxPolyDP(contour, 10, True)
        if len(approx) == 4:
            location = approx
            break
    if (location is None):
        new_text = "Rendszamot nem lehet detektalni"
    
    else:
#---------maszk létrehozása és illesztése-------------------
        mask = np.zeros(gray.shape, np.uint8)
        new_image = cv2.drawContours(mask, [location], 0, 255, -1)
        new_image2 = cv2.drawContours(img.copy(), [location],-1,(0,255,0),2)
        new_image2 = cv2.resize(new_image2,None,fx = 0.6,fy=0.6)
#         cv2.imshow("Rendszamtabla koruli kontur", new_image2)
#         cv2.waitKey(0)
        new_image = cv2.bitwise_and(img, img, mask=mask)
    
#-----------cropping--------------------
        (x,y) = np.where(mask==255)
        (x1, y1) = (np.min(x), np.min(y))
        (x2, y2) = (np.max(x), np.max(y))
        cropped = gray[x1:x2+1, y1:y2+1]

        location = location.reshape(4,2)
        pts = np.array(location, dtype="float32")
        warped = four_point_transform(gray, location)
        warped = cv2.bilateralFilter(warped, 11, 90, 90)             
#         cv2.imshow("Szembe forgatott rendszám",warped)
#         cv2.waitKey(0)        

#----------string kinyerése tesseract-------------

        text = pytesseract.image_to_string(warped, config='--psm 11', lang='eng')
        new_text = text.strip("")
        new_text = ''.join(ch.upper() for ch in new_text if ch.isalnum())
        cv2.destroyAllWindows()
        return new_text

def test_script_different_cars():
    predicted_list = []
    known_plates = []
    i=0
    statistic = []
    with open('teszt.txt') as file:
        lines = file.readlines()
        for line in lines:
            i+=1
            inputpath = line.rstrip()
            img = cv2.imread(inputpath)
            number_plate_text = numberplate_to_text(img)
            actual_plate=get_knownplate(inputpath)
            predicted_list.append(number_plate_text)
            known_plates.append(actual_plate)
            match = round((similar(actual_plate, number_plate_text)*100), 2)
            statistic.append(match)
            print(i,". Exact plate number: ",actual_plate, "   Predicted plate number: ",number_plate_text,"   Match: ",match,"%",'\n')
            
    return statistic       

def punto_test():
    known_plate = "RPS503"
    paths = []
    predicted_list = []
    statistic = []
    for i in range(16):
        temp_string = "punto/"
        temp_string = temp_string+(str(i+1))
        paths.append(temp_string+".jpg")
    for element in paths:
        img = cv2.imread(element)
        number_plate_text = numberplate_to_text(img)
        predicted_list.append(number_plate_text)
        match = round((similar(known_plate, number_plate_text)*100), 2)
        statistic.append(match)
        print("Exact plate number: ",known_plate, "   Predicted plate number: ",number_plate_text,"   Match: ",match,"%",'\n')
    return statistic

statistic = test_script_different_cars()
print("Percentage of 81 detected plates: ",round((sum(statistic)/81), 2), "%.\n\n")

statistic_punto = punto_test()
print("Percentage of 16 different angle: ",round((sum(statistic_punto)/16), 2), "%.\n\n")