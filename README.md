# LincodeFinal

## This code was written for Lincode Hackathon challenge 1

### //////////////////////////////////Cookie Challenge////////////////////////////////////

#### This program (cookiechecker.py) finds the following 
	1. Diameter of the given Cookie 
	2. Number of chocolate chips present	
	3. Damaged cookies, cracked or broken edges
	4. Color of the cookie using top threee hex codes
	 (to check whether the cookie is not burnt) 
 
The diameter of the cookies is found by using the Canny functions of 
open cv to which findes the edges and then dilates them to get fill circle
to polt with. After ploting using Euclidean space, we find the diameter
of the cookies and convert the pixels into inches using the ratio of 
pixelspermetrix. (The values are claibrated denpending on the camera 
and input used)

The top three hex codes are found by traversing the cropped picure pixel 
wise reading all the RGB values and converting them into hex vlaues.
The hex values are only counted if the pixels are in the circle equation.
The avgerage hex color gives the color of the color of the cookie to cross 
check if its burnt or not.

The damaged cookies are checked by a machine learning alogorithm.

The full program can be imported as a package and its functions can 
be used.

Specifications:
hardcoded reference to picture to find diameter
camera: Nikon D3-500
lens: AF-P-NIKKOR 18-55mm
distance from desk: 25.5 cm (using tri pod stand)
Picture dimentions for input: 6,000 x 4,000
