# Railway detection

`Python 3.8` `OpenCv2 4.4.0` `PyQt5`

![Analyzed](https://github.com/grigorevmp/Railway_recognition/blob/master/classic/rw13.jpeg)

In this software I used following algorithms to analyze the railways

* Gray scale conversion
* Gaussian blur through a square kernel of size 11, with standard deviation of 11 pixels in x direction.
* Sobel on the x-axis of the image to make vertical  more detectable. 
* Canny application to obtain the binary image only containing the pixels recognized as edges
* Hough transform to gather the lines of interest.