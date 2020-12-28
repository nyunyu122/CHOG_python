Python code for densely computing local Fourier HOG features, translated from H.Skibbe's original matlab code.


# Overview
Circular Fourier-HOG (or CHOG) features are one of the 2D object detection methods based on Fourier transformed HOG (Histogram of Oriented Gradients). Object detection method using CHOG features was first published in 2012 by H.Skibbe and M.Reisert (see below for more information). Its most important characteristics are 'rotation invariance' and detecting object orientations as well as positions.

CHOG object detection method makes use of supervised learning. To use this method, you need teacher labels (positions and orientations) of destined objects.
You don't have to prepare rotated object labels, though. Given an object label of a particular direction, CHOG can detect the same or similar objects of any other direction as well. (For example, if the input teacher image were an arrow heading to right, CHOG can also detect arrow heading to left, up, down, and other directions. Only one or a few directions are sufficient.) 

Note that even though its detection capability is invariant for rotation, is dependent on contrast and scale (i.e., arrows of different brightness/colors or of different sizes cannot be detected).

From these characteristics, CHOG is suitable for biological microscope images or aerial images, which often include rotated objects of nearly the same size.


# Demos

To run the most simple demo, first download ‘CHOG’ file. It includes all you need to run a demo -- .py scripts (main functions for calculating CHOG features) and cta_demo.ipynb, a teacher image, and a test image.
Just run cta_demo.ipynb on jupyter notebook or on jupyter lab.

Other demo scripts are in the demo_* file.


# Usage

To use this CHOG object detection code, You  have to have teacher image and teacher labels for objects you want to detect. Each teacher label must include object position as x,y-coordinate and direction (object angle) in complex form.


## Object orientation in complex form
If you need to transform object orientation from degree/radian to complex number, see demo_pombe.ipynb.


## Parameter settings
To make the best use of CHOG features, you have to properly set “w_func” parameter, depending on the object size you want to detect.



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

