#Importing the Cookie Program
import CookieChecker
import cv2

#Going thru all 12 images of cookies taken by us, and performing checks
for i in range(1,12):
    image = cv2.imread(str(i)+'.jpg')
    image = cv2.resize(image, None, fx=0.05, fy=0.05, interpolation=cv2.INTER_CUBIC)

    print('Cookie Number:',i)
    print('Number of Chocolate chip cookies:',CookieChecker.countchoco(image))
    diameters = CookieChecker.finddiameter(image)
    print('Horisontal Diameter:', diameters[0])
    print('Vertical Diameter:', diameters[1])
    print('Top three hex codes of the cookie:', CookieChecker.hex_cookie(image))