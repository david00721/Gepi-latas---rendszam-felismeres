import cv2 
import numpy as np 
import imutils
import pytesseract
import string
#import easyocr
from matplotlib import pyplot as plt
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'


#----------rendszám kinyerő függvény-----------

def numberplate_to_text(img):
    cv2.imshow("Rendszamtabla koruli kontur", img)
    cv2.waitKey(0)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    cv2.imshow("Rendszamtabla koruli kontur", gray)
    cv2.waitKey(0)
    bfilter = cv2.bilateralFilter(gray, 11, 17, 17)
    edged = cv2.Canny(bfilter, 100, 300) #Canny algoritmus 30,200, 10, 400
    cv2.imshow("Rendszamtabla koruli kontur", edged)
    cv2.waitKey(0)

#---------------kontúrkeresés-----------------
    
    keypoints = cv2.findContours(edged.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE) 
    contours = imutils.grab_contours(keypoints)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)[:10]

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
        cv2.imshow("Rendszamtabla koruli kontur", new_image2)
        cv2.waitKey(0)
        new_image = cv2.bitwise_and(img, img, mask=mask)
    
#-----------cropping--------------------
        (x,y) = np.where(mask==255)
        (x1, y1) = (np.min(x), np.min(y))
        (x2, y2) = (np.max(x), np.max(y))
        cropped = gray[x1:x2+1, y1:y2+1]
        cv2.imshow("cropped",cropped)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
#----------string kinyerése tesseract-------------
        custom_config = r'--oem 3 --psm 6'
        text = pytesseract.image_to_string(cropped, config = custom_config)
        new_text = text.strip()

#----------string kinyerése easyocr-------------
    
    #reader = easyocr.Reader(['en'])
    #result = reader.readtext(cropped)
    #text = result[0][-2]
    #new_text = text
        cv2.destroyAllWindows()
        return new_text

def teszt_script():
    with open('teszt.txt') as file:
        lines = file.readlines()
        for line in lines:
            inputpath = line.rstrip()
            img = cv2.imread(inputpath)
            number_plate_text = numberplate_to_text(img)
            print(inputpath, "   kép rendszáma:   ",number_plate_text,'\n')

teszt_script()