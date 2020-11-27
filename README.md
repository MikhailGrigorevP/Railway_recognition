# Railway detection

`Python 3.8` `OpenCv2 4.4.0` `PyQt5`

Last improvements : 27.11.2020

![Analyzed](https://github.com/grigorevmp/Railway_recognition/blob/master/classic/rw13.jpeg)

In this software I used following algorithms to analyze the railways

* Contrast adjustment
* Gray scale conversion
* Image binarization

`cv2.threshold(gray, low, high, cv2.THRESH_BINARY)`
* Gaussian blur through a square kernel of size 11, with standard deviation of 11 pixels in x direction.

`cv2.GaussianBlur(image, (kernelSize, kernelSize), 0)`

* Sobel on the x-axis of the image to make vertical  more detectable. 

`cv2.Sobel(image, cv2.CV_8U, 1, 0, ksize=kernel)`

* Canny application to obtain the binary image only containing the pixels recognized as edges

`cv2.Canny(img, low_threshold, high_threshold)`

* Hough transform to gather the lines of interest.

`cv2.HoughLinesP(img, rho, theta, threshold, np.array([]),
                            minLineLength, maxLineGap)`