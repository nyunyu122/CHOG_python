Python code for densely computing local Fourier HOG features, translated from matlab.


# Overview
Circular Fourier-HOG (or CHOG) Features are one of the 2D object detection methods based on Fourier transformed HOG (Histogram of Oriented Gradients). Its most important characteristic is rotation invariance and can detects object orientations as well as positions. 

Once you teach an object of a particular direction, CHOG can detect the same or similar objects of any other direction as well. (For example, if the input teacher image were an arrow heading to right, CHOG can also detect arrow heading to left, up, down, and other directions. As only one or few directions are sufficient, you don’t have to prepare many rotated teacher images.) 
Note that even though its detection capability is invariant for rotation, is dependent on contrast and scale, i.e., arrows of different brightness/color or far different size cannot be detected.

From these characteristics, CHOG is suitable for biological microscope images or aerial images, which often include rotated objects of nearly the same scale.


---------------------
# Original Copyright and disclaimer statement:
Copyright (c) 2011, Henrik Skibbe and Marco Reisert
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met: 

1. Redistributions of source code must retain the above copyright notice, this
   list of conditions and the following disclaimer. 
2. Redistributions in binary form must reproduce the above copyright notice,
   this list of conditions and the following disclaimer in the documentation
   and/or other materials provided with the distribution. 

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR
ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
(INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES 
LOSS OF USE, DATA, OR PROFITS  OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
(INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.


The views and conclusions contained in the software and documentation are those
of the authors and should not be interpreted as representing official policies, 
either expressed or implied, of the FreeBSD Project.

The following code for densely computing local Fourier HOG features
is based on our paper: 
#
Henrik Skibbe and Marco Reisert 
"Circular Fourier-HOG Features for Rotation Invariant Object Detection in Biomedical Images"
in Proceedings of the IEEE International Symposium on Biomedical Imaging 2012 (ISBI 2012), Barcelona 
#
You can find a free copy here :
http://skl220b.ukl.uni-freiburg.de/mr/authoring/mitarbeiter/aktuelle/skibbe/CHOGFilterPaper_en.pdf
#
If you use our functions or partially make use of our code then please cite this paper.

