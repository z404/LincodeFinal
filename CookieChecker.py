'''
------------------------------------------------
PROGRAM TO PERFORM A VISUAL ANALYSIS ON A COOKIE
AND RETURN HORISONTAL AND VERTICAL DIAMETER,
TOP 3 HEX CODES, AND NUMBER OF CHOCOLATE CHIPS
------------------------------------------------
'''

#Importing nessesary packages
from scipy.spatial import distance as dist
from imutils import perspective
from imutils import contours
import numpy as np
import imutils
import cv2
import matplotlib.pyplot as plt
import math
from colormap import rgb2hex, hex2rgb
from sklearn.cluster import KMeans

#Function to find the equation of the circle of the cookie
def circle_equation(x,y,r,l1,image):
    l2=["",""]
    r=r-17
    #l2 list is list of centre of circle
    l2[0] = int((l1[0]+l1[2])/2) #x centre of circle
    l2[1] = int((l1[1]+l1[3])/2) #y centre of circle
    img = cv2.circle(image,(l2[0],l2[1]),int(r/2),(255, 0, 0),1)

    #Inputting values and checking if pixel is inside cookie or not
    if (((x-l2[0])**2 + (y-l2[1])**2)>r**2):
        return 0
    elif(((x-l2[0])**2 + (y-l2[1])**2) <= r**2):
        return 1

#Function wrapper for circle equation
def circle_check(x,y):
    return (circle_equation(x,y,dA,l1,image))

#FUNCTION TO FIND THE TOP 3 HEX COLORS OF THE COOKIE
def hex_cookie(image):

    l,dA,dB = findl(image)
    #Cropping image to just cookie
    crop_img = image[l[0][1]:l[3][1], l[3][0]:l[2][0]]
    dict1 = {}

   #debug function to display histogram
    def make_histogram(cluster):
        numLabels = np.arange(0, len(np.unique(cluster.labels_)) + 1)
        hist, _ = np.histogram(cluster.labels_, bins=numLabels)
        hist = hist.astype('float32')
        hist /= hist.sum()
        return hist

   #debug function to display hex bar color
    def make_bar(height, width, color):
        bar = np.zeros((height, width, 3), np.uint8)
        bar[:] = color
        red, green, blue = int(color[2]), int(color[1]), int(color[0])
        hsv_bar = cv2.cvtColor(bar, cv2.COLOR_BGR2HSV)
        hue, sat, val = hsv_bar[0][0]
        return bar, (red, green, blue), (hue, sat, val)

    #debug function to sort hsv values and showing ordered bar
    def sort_hsvs(hsv_list):
        bars_with_indexes = []
        for index, hsv_val in enumerate(hsv_list):
            bars_with_indexes.append((index, hsv_val[0], hsv_val[1], hsv_val[2]))
        bars_with_indexes.sort(key=lambda elem: (elem[1], elem[2], elem[3]))
        return [item[0] for item in bars_with_indexes]

    #dimentions of cropped image
    height, width, _ = np.shape(crop_img)

    #reshape the image to be a simple list of RGB pixels
    image = crop_img.reshape((height * width, 3))

    #picking the 5 most common colors
    num_clusters = 5
    clusters = KMeans(n_clusters=num_clusters)
    clusters.fit(image)

    #count the dominant colors and put them in clusters
    histogram = make_histogram(clusters)
   
    #sorting them, most-common first
    combined = zip(histogram, clusters.cluster_centers_)
    combined = sorted(combined, key=lambda x: x[0], reverse=True)

    #finally, we'll output a graphic showing the colors in order
    bars = []
    hsv_values = []

    #list of top three rgbs in picture
    top3 = []

    #gong thru clusters
    for index, rows in enumerate(combined):
        #getting bar, rgb, and hsv
        bar, rgb, hsv = make_bar(100, 100, rows[1])
        hsv_values.append(hsv)
        #debug list for displaying bar
        bars.append(bar)

        #ignoring very light colors that make up background
        if not(rgb[0]>150 and rgb[1]>150 and rgb[1]>150):
            top3.append(rgb)
            if len(top3) == 3:
                break

    #converting rgb to hex
    top3hex = []

    #iterating thru rgb values
    for i in top3:
        r,g,b = i
        top3hex.append(rgb2hex(r,g,b))

    #returning top 3 hexes
    return top3hex


#Funtion to return midpoint of box
def midpoint(ptA, ptB):
    return ((ptA[0] + ptB[0]) * 0.5, (ptA[1] + ptB[1]) * 0.5)


  
#Funtion to find the region of interest of the picture
def region_interest(image):
    height = image.shape[0]
    polygons = np.array([
        [(10, 200), (400,200), (400,10), (10,10)]
        ])
    mask = np.zeros_like(image)
    cv2.fillPoly(mask,polygons,255)
    masked_image = cv2.bitwise_and(image,mask)
    return masked_image

#FUNCTION TO FIND DIAMETER AND TOP 3 HEX OF COOKIE
def findl(image):

    #Function to make edges 
    def canny(image):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (5, 5), 0)
        canny = cv2.Canny(blur, 50,150)
        return canny
    
    image2 = np.copy(image)
    canny = canny(image2)
    cropped_image = region_interest(canny)
    edged = cv2.dilate(cropped_image, None, iterations=5)
    edged = cv2.erode(edged, None, iterations=0)

    #find contours
    cnts = cv2.findContours(edged.copy(), cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)

    #sort the contours from left-to-right and initialize the
    #'pixels per metric' calibration variable
    (cnts, _) = contours.sort_contours(cnts)

    #List to maintain all points of a rectangle
    l=[]
    #iterating thru contours
    for c in cnts:
        if cv2.contourArea(c) < 2000:
            continue
        #Compute the rotated bounding box of the contour
        orig = image.copy()
        box = cv2.minAreaRect(c)
        box = cv2.cv.BoxPoints(box) if imutils.is_cv2() else cv2.boxPoints(box)
        box = np.array(box, dtype="int")
        #order the points in the contour such that they appear
        #in top-left, top-right, bottom-right, and bottom-left
        #order, then draw the outline of the rotated bounding box
        box = perspective.order_points(box)
        cv2.drawContours(orig, [box.astype("int")], -1, (0, 255, 0), 2)
        # loop over the original points and draw them
        for (x, y) in box:
            cv2.circle(orig, (int(x), int(y)), 5, (0, 0, 255), -1)
            l.append([int(x), int(y)])

        # unpack the ordered bounding box, then compute the midpoint
        # between the top-left and top-right coordinates, followed by
        # the midpoint between bottom-left and bottom-right coordinates
        (tl, tr, br, bl) = box
        (tltrX, tltrY) = midpoint(tl, tr)
        (blbrX, blbrY) = midpoint(bl, br)
        # compute the midpoint between the top-left and top-right points,
        # followed by the midpoint between the top-righ and bottom-right
        (tlblX, tlblY) = midpoint(tl, bl)
        (trbrX, trbrY) = midpoint(tr, br)
        # draw the midpoints on the image
        cv2.circle(orig, (int(tltrX), int(tltrY)), 5, (255, 0, 0), -1)
        cv2.circle(orig, (int(blbrX), int(blbrY)), 5, (255, 0, 0), -1)
        cv2.circle(orig, (int(tlblX), int(tlblY)), 5, (255, 0, 0), -1)
        cv2.circle(orig, (int(trbrX), int(trbrY)), 5, (255, 0, 0), -1)

        cv2.line(orig, (int(tltrX), int(tltrY)), (int(blbrX), int(blbrY)),(255, 0, 255), 2)
        cv2.line(orig, (int(tlblX), int(tlblY)), (int(trbrX), int(trbrY)),(255, 0, 255), 2)

        #Pixels of length and width of cookie
        dA = dist.euclidean((tltrX, tltrY), (blbrX, blbrY))
        dB = dist.euclidean((tlblX, tlblY), (trbrX, trbrY))
        break
    
    return l,dA,dB #returning L, Pixels (x axis, y axis)

def finddiameter(image):
    l,dA,dB = findl(image)
    #return all details
    
    #hardcoded reference to picture to find diameter
    #camera: Nikon D3-500
    #lens: AF-P-NIKKOR 18-55mm
    #distance from desk: 25.5 cm
    width = 8.1
    #getting ratio width/pixels
    ratio = width/image.shape[0]
    
    return dB*ratio*0.393,dA*ratio*0.393 #return horisontal diameter, return vertical diameter
    

#FUNCION TO COUNT NUMBER OF CHOCOLATE CHIPS THAT ARE VISIBLE IN THE COOKIE
def countchoco(original):
    '''
    Funtion to find number of chocolate chip cookies
    '''
    #converting to black and white
    gray_im = cv2.cvtColor(original, cv2.COLOR_BGR2GRAY)
    #Contrast adjusting with gamma correction y = 0.07
    gray_correct = np.array(255 * (gray_im / 255) ** .07 , dtype='uint8')
    
    #Contrast adjusting with histogramm equalization
    gray_equ = cv2.equalizeHist(gray_im)
    
    #Local adaptative threshold
    lower = np.array([140])  #-- Lower range --
    upper = np.array([256])  #-- Upper range --
    mask = cv2.inRange(gray_correct, lower, upper)
    res = cv2.bitwise_and(gray_correct, gray_correct, mask= mask) #-- Contains pixels having the gray color--

    gray_correct = res
    thresh = cv2.adaptiveThreshold(gray_correct, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 255, 20)
    
    #Dilatation and erosion
    kernel = np.ones((1, 4), np.uint8)
    img_dilation = cv2.dilate(thresh, kernel, iterations=1)
    img_erode = cv2.erode(img_dilation, kernel, iterations=1)
    
    #clean all noise after dilatation and erosion
    img_erode = cv2.medianBlur(img_erode, 1)
    #plt.subplot(221)
    #plt.title('Dilatation + erosion')
    #plt.imshow(img_erode, cmap="gray", vmin=0, vmax=255)

    img_not = cv2.bitwise_not(img_erode)
    img_not = cv2.GaussianBlur(img_not,(5,5),3)
    #cv2.imshow("Invert1", img_not)

    img_erode = img_not
    #plt.show()

    #Labeling
    ret, labels = cv2.connectedComponents(img_erode)
    label_hue = np.uint8(179 * labels / np.max(labels))
    blank_ch = 255 * np.ones_like(label_hue)
    labeled_img = cv2.merge([label_hue, blank_ch, blank_ch])
    labeled_img = cv2.cvtColor(labeled_img, cv2.COLOR_HSV2BGR)
    labeled_img[label_hue == 0] = 0

    #plt.subplot(222)
    #plt.title('Objects counted:'+ str(ret-1))
    #plt.imshow(labeled_img)
    #print('objects number is:', ret-1)
    #plt.show()
    
    #return number of chips
    return ret-1

def performChecks(image,minchocochips=5):
    '''
    Checks weather the cookie matches the specifications given
    '''

    returnlst = {}
    
    hori,verti = finddiameter(image)
    if abs(hori-verti) > 1: returnlst.update({'diameter':'False'})
    else: returnlst.update({'diameter':'True'})

    countofchoco = countchoco(image)
    if countofchoco < minchocochips: returnlst.update({'numberofchips':'False'})
    else: returnlst.update({'numberofchips':'True'})

    hexcodes = hex_cookie(image)
    flag = True
    for i in hexcodes:
        r,g,b = hex2rgb(i)
        if r<40 and g<40 and b<40:
            flag = False
            break
    if flag: returnlst.update({'color':'True'})
    else: returnlst.update({'color':'False'})

    return returnlst
#START HERE
#image = cv2.imread("2.jpg")
#image = cv2.resize(image, None, fx=0.05, fy=0.05, interpolation=cv2.INTER_CUBIC)

#diametertuple = finddandhex(image)
#print(finddiameter(image))
#print(hex_cookie(image))
#print(countchoco(image))
