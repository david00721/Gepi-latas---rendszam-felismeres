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
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)    
    cv2.imshow("Kiindulo kep", img)
    cv2.imshow("Szurke kep", gray)
    bfilter = cv2.bilateralFilter(gray, 11, 17, 17) #zajcsökkentés
    edged = cv2.Canny(bfilter, 10, 210) #Canny algoritmus 30,200, 10, 400
    korvonalas_kep=cv2.cvtColor(edged, cv2.COLOR_BGR2RGB)
    cv2.imshow("Korvonalazott kep", korvonalas_kep)

#---------------kontúrkeresés-----------------
    
    keypoints = cv2.findContours(edged.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE) 
    contours = imutils.grab_contours(keypoints)
    contour_kiiratas =cv2.drawContours(img.copy(), contours, -1, (0,255,0),1)
    cv2.imshow("Konturok", contour_kiiratas)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)[:10] # rendezi és visszaadja az első 10-et

#----------------rendszám megkeresése-----------------------
    location = None
    for contour in contours: #ugye rendeztük korábban
        approx = cv2.approxPolyDP(contour, 10, True) #approximate polygon from contour, közelíti a poligont a kontúrbol, minel nagyobb a középső argumentum, annál jobban kezeli egyenesnek a "recés részeket"
        if len(approx) == 4:
            location = approx #ha megvan a 4 sarokpont akkor elmenti és kilép
            break

#---------maszk létrehozása és illesztése-------------------
        
    mask = np.zeros(gray.shape, np.uint8) #maszk
    new_image = cv2.drawContours(mask, [location], 0,255, -1) # létrehozza az ideiglenes képet ami a korábban számított sarokpontok között van (location)
    new_image2 = cv2.drawContours(img.copy(), [location],-1,(0,255,0),2)#kirajzoláshoz kell
    cv2.imshow("Rendszamtabla koruli kontur", new_image2)
    new_image = cv2.bitwise_and(img, img, mask=mask) # az ideiglenes képre amin a maszk van berakja az eredeti képet, így csak a rendszámtábla látszik 
    
#-----------cropping--------------------
    
    (x,y) = np.where(mask==255)
    (x1, y1) = (np.min(x), np.min(y))
    (x2, y2) = (np.max(x), np.max(y))
    cropped = gray[x1:x2+1, y1:y2+1]
    cv2.imshow("1. lépés", cropped)

#----------string kinyerése tesseract-------------
    
    text = pytesseract.image_to_string(cropped, config='--psm 11')
    new_text = text.strip()

#----------string kinyerése easyocr-------------
    
    #reader = easyocr.Reader(['en'])
    #result = reader.readtext(cropped)
    #text = result[0][-2]
    
    return new_text
   
inputpath = input('Adja meg a kép elérési útját! (formátum: tesztkepek/ .. .jpg)\n')
img = cv2.imread(inputpath) #kép beolvasása fájlba
number_plate_text = numberplate_to_text(img)
print("A rendszam:",number_plate_text)
cv2.waitKey()
cv2.destroyAllWindows()