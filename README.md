 

<script src="https://cdnjs.cloudflare.com/ajax/libs/mathjax/2.7.5/MathJax.js?config=TeX-MML-AM_CHTML"></script>

# INDEPENDENT STUDY COURSE

## Abstract 
**Ravichandra Pothamsetty** completed this work as part of an Independent Study Course for one credit, under the guidance of **Dr.David Crandall**. The project aims to implement an existing method for calculating the depth of a scene using two images of the scence taken by the same camera. The process is implemented in python from scratch.The problem can be tackled in two stages: first, calibrating the camera parameters, and second, estimating the depth of individual pixels in the image. 
## Introduction
Generating a 3D image is crucial for its wide range of applications, such as in medical imaging, virtual reality, and autonomous systems. When a camera captures images of the world, it produces a 2D representation of the scene. However, since the world is a 3D space and the image plane is 2D, important information may be lost during this process. By utilizing multiple images, we can estimate the relative positions of objects within the scene. Personally, I was fascinated by the idea of using basic linear algebra and geometry concepts to estimate the depth of an image when I first learned about it in a computer vision course. This motivated me to delve deeper and study the step-by-step methods for estimating image depth.
<p>The same concept has came up in Autonomous systems Course where cameras are  used as sensors in autonomous vehicles to estimate depth. By analyzing the images captured by the camera, the vehicle can determine the distance to objects in its surroundings and construct a three-dimensional map of its environment.This information can be combined with data from other sensors to create a more comprehensive and accurate picture of the vehicle's surroundings, enabling it to navigate and make decisions autonomously. </p>

## Problem Statement ##
Given two images _I<sub>1</sub>_ and _I<sub>2</sub>_ of the scence _S_ captured by using the Camera _C_ , estimate the depth of each pixel in the image(either in _I<sub>1</sub>_ or _I<sub>2</sub>_). The image _I<sub>1</sub>_ and _I<sub>2</sub>_ are more less the same just that camera position and orientation differ when capturing the images.

## Approach
1. __Camera Calibration__: We can assume a linear model relationship between the 3 Dimensional point in the scene and the 2 Dimensional pixel cordinates in the image.Linear model makes things easy with less number of parameters. There are two parts to the linear model , first the position & orientation of the camera which are called the externsic parameters of the camera, second is the how the camera projects the 3 Dimensional point on to 2 dimensional image plane called internsic parameters of the camera. The estimation of the intrinsic and externsic parameters of the camera is called camera calibration. For any give camera , the intrinsic parameters remains constant whereas the externsic parameters keeps on changing from image to image(as we change the positon & orientation of the camera). For the depth estimation purpose , we just need the intrinsic parameters of the camera. 
Let us assume a Point __P__ in the Scene with coordinates (x<sub>w</sub>, y<sub>w</sub>, z<sub>w</sub>) with respect to world coordinate system and (x<sub>c</sub>, y<sub>c</sub>, z<sub>c</sub>) with respect to the camera with origin at pinhole of the camera.Let (x<sub>i</sub>, y<sub>i</sub>) be the projection of __P__ on to the image plane using the camera. Using the similar triangles property from the geometry

<p align="center" width="100%">
<img src=similar-triangles.png width=30% height=10%>
</p>

where $f$ is the focal length of the camera lens
$$ \begin{equation} \frac{x_i}{f} = \frac{x_c}{z_c} \end{equation}$$
Rearraning the terms would lead to the following equation
$$\begin{equation} x_i = \frac{f \cdot x_c}{z_c} \end{equation}$$
The camera reference system might not coincide with digital image coordinate system. Typically we take one corner of the image as the center of the image . So we need to accomodate for the translation. Hence the equation becomes 

$$\begin{equation} x_i = \frac{f \cdot x_c}{z_c}+c_x \end{equation}$$

$$\begin{equation} y_i = \frac{f \cdot y_c}{z_c}+c_y \end{equation}$$

The camera reference system measures everything in centi meters or millimeters or some physical measurement. But in the digital image everything is in pixels.To adjust for that we multiply with constant $k$ which is measure of number of pixels per unit length. So the equation now becomes 
$$\begin{equation} x_i = k \cdot \frac{ f \cdot x_c}{z_c}+c_x \end{equation}$$
Similarily in the y direction as well
$$\begin{equation} y_i = l \cdot \frac{ f \cdot y_c}{z_c}+c_y \end{equation}$$
where $l$ is the number of pixels per unit length in $y$ direction.<br>
We can simiplify both the above equations as 
$$\begin{equation} x_i = \frac{ f_x \cdot x_c}{z_c}+c_x \end{equation}$$
$$\begin{equation} y_i = \frac{ f_y \cdot y_c}{z_c}+c_y\end{equation}$$

With $z_c$ in the denominator the above equations are in non linear fashion. Using homogenous coordinate , we can represent them in linear model. That is ( $x_i$ , $y_i$ , $z_i$) maps to ( $x_c$ , $y_c$, $z_c$, $1$ ). Here $z_i$ is $z_c$. Representing in the matrix form

$$\begin{equation}
\begin{bmatrix}

f_x & 0 & c_x & 0 \\
0 & f_y & c_y & 0 \\
0 & 0 &1 & 0 \\
\end{bmatrix}

\begin{bmatrix}
x_c \\
y_c \\
z_c \\
1 \\
\end{bmatrix}

=
\begin{bmatrix}
x_i \\
y_i \\
z_i \\
\end{bmatrix}
\end{equation}$$


But in reality we will be having points in the scene with respect to world coordinate system. Since any transformation can be obtained combination of translation and rotation we can write as follows where $R$ is the rotation matrix and $t$ is the translational vector. 
$$ \begin{equation}P_c = RP_w + t \end{equation}$$
which effectively is

$$
\begin{equation}
\begin{bmatrix}
r_{11} & r_{12} & r_{13} & t_x \\
r_{21} & r_{22} & r_{23} & t_y\\
r_{31} & r_{32} & r_{33} & t_z \\
0 & 0 & 0 & 1\\
\end{bmatrix}
\begin{bmatrix}
x_w \\
y_w \\
z_w \\
1 \\
\end{bmatrix}

=
\begin{bmatrix}
x_c \\
y_c \\
z_c \\
1 \\
\end{bmatrix}
\end{equation}
$$ 

Combining  equations 9 and 11 
$$
\begin{equation}
\begin{bmatrix}
f_x & 0 & c_x & 0 \\
0 & f_y & c_y & 0 \\
0 & 0 &1 & 0 \\
\end{bmatrix}
\begin{bmatrix}
r_{11} & r_{12} & r_{13} & t_x \\
r_{21} & r_{22} & r_{23} & t_y\\
r_{31} & r_{32} & r_{33} & t_z \\
0 & 0 & 0 & 1\\
\end{bmatrix}
\begin{bmatrix}
x_w \\
y_w \\
z_w \\
1 \\
\end{bmatrix}

=
\begin{bmatrix}
x_i \\
y_i \\
1 \\
\end{bmatrix}
\end{equation}
$$ 
The above equations can be rewritten as following 
$$\begin{equation} P_i = M \cdot P_w \end{equation}$$ 
where $M$ is called the _projection-matrix_
$$
\begin{equation}
\begin{bmatrix}
x_i \\
y_i \\
1 \\
\end{bmatrix}

=
\begin{bmatrix}
m_{11} & m_{12} & m_{13} & m_{14} \\
m_{21} & m_{22} & m_{23} & m_{24} \\
m_{31} & m_{32} & m_{33} & m_{34} \\
\end{bmatrix}
\begin{bmatrix}
x_w \\
y_w \\
z_w \\
1 \\
\end{bmatrix}
\end{equation}
$$
$$\begin{equation} x_i=\frac{m_{11}x_w+m_{12}y_w+m_{13}z_w+m_{14}}{m_{31}x_w+m_{32}y_w+m_{33}z_w+m_{34}} \end{equation}$$
$$\begin{equation} y_i=\frac{m_{21}x_w+m_{22}y_w+m_{23}z_w+m_{24}}{m_{31}x_w+m_{32}y_w+m_{33}z_w+m_{34}} \end{equation}$$
The above equation can be rearranged as 
$$ 
\begin{equation}
\begin{bmatrix}
x_w & y_w & z_w & 1 & 0 & 0 & 0 & 0 & -x_i x_w & -x_i y_w & -x_i z_w & -x_i \\
0 & 0 & 0 & 0 & x_w & y_w & z_w & 1 & -y_i x_w & -y_i y_w & -y_i z_w & -y_i \\
\end{bmatrix}


\begin{bmatrix}
m_{11} \\
m_{12} \\
m_{13} \\
m_{14} \\
m_{21} \\
m_{22} \\
m_{23} \\
m_{24} \\
m_{31} \\
m_{32} \\
m_{33} \\
m_{34} \\
\end{bmatrix}

=
\begin{bmatrix}
0 \\
0 \\
0 \\
0 \\
0 \\
0 \\
0 \\
0 \\
0 \\
0 \\
0 \\
0 \\
\end{bmatrix}
\end{equation}
$$
If we have  multiple pairs of correspondences then equation will be of the same above except that the first matrix will have more number of rows . Let us assume first matrix is called $A$. Subsequently the equation will looks of the form 
$$\begin{equation}AM=0\end{equation}$$
For mulitple pair of  correspondences between the world coordinates and the pixel coordinates this becomes as a system of overdetermined equations. We can ignore the trivial solution as it is not useful and pose this as an optimization problem i.e, minimize AM such that L-2 norm of M is one.<br>
The solution to this problem is the eigen vector corresponding to the least eigen value of A<sup>T</sup>A. Reshaping the M-vector from this solution into 3x4 matrix will gives the projection matrix<br>
A simple [QR-decomposition](https://en.wikipedia.org/wiki/QR_decomposition) of first three columns of $M$ will be the intrinsic parameters of the camera and the rotation matrix<br>
The translation vector $t$ can be found by using the 
$$\begin{equation}t=Inverse(K) \cdot [m_{14} \ m_{24} \ m_{34}]^T\end{equation}$$


## Depth Estimation: 
Depth estimation makes use of the epipolar constraint. Let say the $P$ be a point in the scence and $O1$ and $O2$ are the projections of $P$ using Camera $C_l$ and $C_r$ respectively. Then the points $O1$, $O2$, $P$ lie on a plane called __epipolar-line__ The projections on the image plane of one camera must lie on a line called the __epipolar line__.<br>
In depth estimation or stereo vision one camera is called left camera and other is called right camera. Let say there are cameras $C_l$, $C_r$ captured a point $P$ in the scence. Let $R$ $t$ be the rotational matrix & translational vector to obtain $C_r$ cordinate reference system with respect to $C_l$ coordinate system. If $P$ has coordinates $P_{lc}$ ( $x_l$, $y_l$, $z_l$) in the left camera's coordinate system and $P$ has coordinates $P_{rc}$ ($x_r$, $y_r$, $z_r$) in the  right camera's coordinate system.
$$\begin{equation}P_{lc} = R\cdot P_{rc} + t\end{equation}$$
We can have another constaint
$$\begin{equation}P_{lc} \cdot (t \times P_{lc}) = 0\end{equation}$$
The above constraint is simple as it is the dot product of a vector which is perpendicular to the vector containing itself. Substituting the $P_{lc}$ from equation 20 in one of the $P_{lc}$ in equation 21 which results
$$\begin{equation}P_{lc} \cdot (t \times (R\cdot P_{rc} + t))=0\end{equation}$$
reduces to
$$\begin{equation}P_{lc} \cdot (t \times R\cdot P_{rc})=0\end{equation}$$ 
Let $E= t \times R$
$$\begin{equation}P_{lc} \cdot E \cdot P_{rc} = 0\end{equation}$$
So the epipolar constraint boils down to 
$$\begin{equation}
\begin{bmatrix}
x_{l} &
y_{l} &
z_{l} \\
\end{bmatrix}
\cdot
\begin{bmatrix}
e_{11} & e_{12} & e_{13} \\
e_{21} & e_{22} & e_{23} \\
e_{31} & e_{32} & e_{33} \\
\end{bmatrix}
\cdot
\begin{bmatrix}
x_{r} \\
y_{r} \\
z_{r} \\
\end{bmatrix}

=
0
\end{equation}
$$
Where $E$ is called the __essential\_matrix__ . The problem is we do not have $P_{lc}$ and $P_{rc}$ but we have the pixel coordinates. Since we already have the relation between pixel coordinates and camera reference coordinates . We can substitute that in the equation 25.
$$\begin{equation}P_{li}=K \cdot P_{lc}\end{equation}$$
where $P_{li}$ is left image pixel coordinate system and $K$ is the intinsic parameters matrix of the camera. We can assume that both left camera and right camera are the same.
Substituting the in the equation 25, we will arrive at
$$\begin{equation}P_{li} \cdot K^{-1 \ T} \cdot E \cdot K^{-1} \cdot P_{ri} = 0\end{equation}$$
Let $K^{-1 \ T} \cdot E \cdot K^{-1}=F$, then we can rewrite the equation as 
$$\begin{equation}P_{li} \cdot F \cdot P_{ri}=0\end{equation}$$
where $F$ is fundamental matrix of  $3 \times 3$ size.
The objective is to find the depth which is either $z_{l}$ or $z_{r}$ for each pixel. These are the steps we follow<br>
1. Estimate the fundamental Matrix using the correspondence pairs between two images. To get the correspondence , we can make use of the interesting point extractor (ORB) from OpenCV library.
2. From fundamental matrix, compute the essential matrix.
3. By applying Singular Value Decomposition of the  essential matrix , we can find the rotational and translational matrices of the stereo. 
4. Using the Fundamental matrix , for each pixel in the left image, one can find the epipolar line equation in the right image. Among all the pixels on the epipolar line, we will match with the point which has intensity closer to the pixel in the left image using least square error as the metric. Now for each pixel in the left image , we have corresponding pixel in the right image.
5. We have equation 29(camera to pixel coordinates) and equation 30(matrix form of equation 20).

$$\begin{equation}P_{ri}= K \cdot P_{rc}\end{equation}$$
$$
\begin{equation}
P_{lc}

=
\begin{bmatrix}
R & t\\
0 & 1\\
\end{bmatrix}
\cdot P_{rc}
\end{equation}
$$

Multiply equation 30 with $K$ matrix on both sides, we arrive at 

$$
\begin{equation}
P_{li}

=
K
\cdot
\begin{bmatrix}
R & t\\
0 & 1\\
\end{bmatrix}
\cdot P_{rc}
\end{equation}
$$
From equation 31 , we can write 
$$
\begin{equation}
P_{li} = M_{stereo} \cdot P_{rc}
\end{equation}
$$
where $M_{stereo}$ is the equivalent of projection matrix for the stereo.<br>
Using equation 29 and equation 32 , we can solve for $P_{rc}$  since $M_{stereo}$, $P_{ri}$, $P_{li}$, $K$ are known.<br> 
Once $P_{rc}=[x_{rc} \ y_{rc} \ z_{rc}]$ is known, depth is nothing but $z_{rc}$. Similar calculation can be done for $P_{lc}$ and to arrive at $z_{lc}$ depth in the left image 

## Code 
The code should be run<br>
```python depth-estimation-project.py data-set-directory-name```<br>
We assume the data-set-directory-name has following structure. Here im0.png , im1.png represents the images of the same scene and calib.txt file contains the intrinsic parameters of the camera with which the two images are taken
.
 * [data-set-directory-name](./tree-md)
    * [folder1](./folder1)
        * [im0.png](./folder1/im0.png)
        * [im1.png](./folder1/im1.png)
        * [calib.txt](./folder1/calib.txt)
    * [folder2](./folder2)
        * [im0.png](./folder2/im0.png)
        * [im1.png](./folder2/im1.png)
        * [calib.txt](./folder2/calib.txt)
    * [folder3](./folder3)
        * [im0.png](./folder3/im0.png)
        * [im1.png](./folder3/im1.png)
        * [calib.txt](./folder3/calib.txt)

The output of the program will be written to ```data-set-directory-name-results``` directory. Each folder will have the key points image, keypoints matching image, epipolar lines image and the disparity image. Using the [joblib](https://joblib.readthedocs.io/en/latest/index.html) library to run the different datasets in parallel<br>

## Stereo-Results-Datasets
Picked a few  stereo datasets from [middlebury-stereo-dataset](https://vision.middlebury.edu/stereo/data/scenes2021/)  which has both the left and right images of the scene and the camera calibiration parameters.<br>

### Example-1

Left & Right Image<br>

<img src=mobile-stereo-datasets-results/Pendulum1/key-points-1.png width=50% height =50% title="Title"><img src=mobile-stereo-datasets-results/Pendulum1/key-points-2.png width=50% height =50% title="Title">

Visualizing the Key Points Matching<br>

<img src=mobile-stereo-datasets-results/Pendulum1/matching-key-points.png title="Title">

Visualizing the Epipolar Lines <br>
<img src=mobile-stereo-datasets-results/Pendulum1/epipolar_lines.png title="Title">

Disparity Image <br>

<img src=mobile-stereo-datasets-results/Pendulum1/disparity.jpeg title="Title">

### Example-2

Left & Right Image<br>

<img src=mobile-stereo-datasets-results/chess1/key-points-1.png width=50% height =50% title="Title"><img src=mobile-stereo-datasets-results/chess1/key-points-2.png width=50% height =50% title="Title">

Visualizing the Key Points Matching<br>

<img src=mobile-stereo-datasets-results/chess1/matching-key-points.png title="Title">

Visualizing the Epipolar Lines <br>
<img src=mobile-stereo-datasets-results/chess1/epipolar_lines.png title="Title">

Disparity Image <br>

<img src=mobile-stereo-datasets-results/chess1/disparity.jpeg title="Title">


### Example-3

Left & Right Image<br>

<img src=mobile-stereo-datasets-results/art-room1/key-points-1.png width=50% height =50% title="Title"><img src=mobile-stereo-datasets-results/art-room1/key-points-2.png width=50% height =50% title="Title">

Visualizing the Key Points Matching<br>

<img src=mobile-stereo-datasets-results/art-room1/matching-key-points.png title="Title">

Visualizing the Epipolar Lines <br>
<img src=mobile-stereo-datasets-results/art-room1/epipolar_lines.png title="Title">

Disparity Image <br>

<img src=mobile-stereo-datasets-results/art-room1/disparity.jpeg title="Title">


### Example-4

Left & Right Image<br>

<img src=mobile-stereo-datasets-results/Podium/key-points-1.png width=50% height =50% title="Title"><img src=mobile-stereo-datasets-results/Podium/key-points-2.png width=50% height =50% title="Title">

Visualizing the Key Points Matching<br>

<img src=mobile-stereo-datasets-results/Podium/matching-key-points.png title="Title">

Visualizing the Epipolar Lines <br>
<img src=mobile-stereo-datasets-results/Podium/epipolar_lines.png title="Title">

Disparity Image <br>

<img src=mobile-stereo-datasets-results/Podium/disparity.jpeg  title="Title">


### Example-5

Left & Right Image<br>

<img src=mobile-stereo-datasets-results/skates2/key-points-1.png width=50% height =50% title="Title"><img src=mobile-stereo-datasets-results/skates2/key-points-2.png width=50% height =50% title="Title">

Visualizing the Key Points Matching<br>

<img src=mobile-stereo-datasets-results/skates2/matching-key-points.png title="Title">

Visualizing the Epipolar Lines <br>
<img src=mobile-stereo-datasets-results/skates2/epipolar_lines.png title="Title">

Disparity Image <br>

<img src=mobile-stereo-datasets-results/skates2/disparity.jpeg title="Title">


### Example-6

Left & Right Image<br>

<img src=mobile-stereo-datasets-results/cuerule1/key-points-1.png width=50% height =50% title="Title"><img src=mobile-stereo-datasets-results/cuerule1/key-points-2.png width=50% height =50% title="Title">

Visualizing the Key Points Matching<br>

<img src=mobile-stereo-datasets-results/cuerule1/matching-key-points.png title="Title">

Visualizing the Epipolar Lines <br>
<img src=mobile-stereo-datasets-results/cuerule1/epipolar_lines.png title="Title">

Disparity Image <br>

<img src=mobile-stereo-datasets-results/cuerule1/disparity.jpeg  title="Title">


## Stereo-Results-SmartPhone-Camera
Previously we tried with datasets with two images and given calibration. Now we would like to try with smartphone cameras.For this purpose , the first step is to calibrate the camera inorder to get the camera parameters and next is compute the depth. To estimate the intrinsic parameters of the camera, we need few points of correspondence between the world points and the pixel coordinates. For this purpose we checker-board with known measurements and take its image with camera.

<img src=PatternEdit.jpeg width=50% height=30%>

I have taken the printout, considered few corner points(where squares intersect) and find their pixel coordinates using this code. Since we know the length of side of  square, by taking one of the corner point as origin for the world coordinate system, we can find the coordinates any point(by adding or subtracting the length of side of the squares). Once we have correspondence pairs between the world coordinates and pixel coordinates, we can construct the $A$ matrix mentioned in camera-calibration section and solve for the projection matrix $M$. We deduce the intrinsic parameters as discussed in the camera calibration seciton.<br>

The only issue is that all the points should __NOT__ be coplanar. Initially I have taken few points on the paper pattern(taken the image placed on surface).So the $z$ coordinate is zero(or same) for all points.Hence they are coplanar. When I had run using these points the smallest eigen value of  $A^T \cdot A$ matrix ended up zero which made impossible to compute the parameters of the camera. <br>

Next considered a mulitple of above patterns which can form a cube shape as show below.Selection of  points are done so that all points are not coplanar <br>

<img src=3dPattern.png width=50% height=50%>

Patterns I used to calibirate the phone camera parameters. I have attached the pattern printouts on the sides of cardboard box(sides are orthogonal)<br>
<img src=samsung-phone-pattern/pattern1.jpeg width=30% height=30%><img src=samsung-phone-pattern/Pattern2.jpeg width=30% height=30%> <br>

Used the code in the file ``` get-pixel-coordinates.py ``` to get the pixel coordinates using interactive method. Click on the image and the pixel coordinates will be displayed on the command line<br>

Running the file ```camera-calibration-project.py``` will read lists from the  ```corresponding-pairs.yaml``` file and will produce camera parameters in the ```calib.txt``` file . The yaml file contains the list of world coordinates and pixel coordinates.<br>

Running the ```depth-estimation-project.py``` with images take by this camera will yield the results 

### Example-1

Left & Right Image<br>

<img src=mobile-stereo-datasets-results/phone-camera-cup/key-points-1.png width=50% height =50% title="Title"><img src=mobile-stereo-datasets-results/phone-camera-cup/key-points-2.png width=50% height =50% title="Title">

Visualizing the Key Points Matching<br>

<img src=mobile-stereo-datasets-results/phone-camera-cup/matching-key-points.png title="Title">

Visualizing the Epipolar Lines <br>
<img src=mobile-stereo-datasets-results/phone-camera-cup/epipolar_lines.png title="Title">

Disparity Image <br>

<img src=mobile-stereo-datasets-results/phone-camera-cup/disparity.jpeg width=50% height =50% title="Title">

### Example-2

Left & Right Image<br>

<img src=mobile-stereo-datasets-results/phone-camera-table/key-points-1.png width=50% height =50% title="Title"><img src=mobile-stereo-datasets-results/phone-camera-table/key-points-2.png width=50% height =50% title="Title">

Visualizing the Key Points Matching<br>

<img src=mobile-stereo-datasets-results/phone-camera-table/matching-key-points.png title="Title">

Visualizing the Epipolar Lines <br>
<img src=mobile-stereo-datasets-results/phone-camera-table/epipolar_lines.png title="Title">

Disparity Image <br>

<img src=mobile-stereo-datasets-results/phone-camera-table/disparity.jpeg width=50% height =50% title="Title">


In the second example , there is little bit shape of the  bottle and glass. The results are as good as that of middlebury datasets because may be the linear assumption of the camera model is not accurate for these pictures and hence estimated camera model is not working which resulted not so good disparity images

## Learnings

Through this , I have learned  what are the assumptions and how mathematical model of camera is developed. Coming to the stereo, developed understanding about the epipolar constraint and how this constraint lays foundation for depth estimation. Along with this by the virtue of the project, I had revisited the Linear Algebra concepts such as Eigen Vectors, Singular Value Decomposition etc and  constrained optimization ie how to maximize and minimize when you have bunch of constraints<br>

## References

1. https://www.tutorialspoint.com/opencv-python-how-to-display-the-coordinates-of-points-clicked-on-an-image
2. https://www.researchgate.net/figure/The-similar-triangle-relationship-in-the-pinhole-camera-model_fig2_350330292
3. https://www.youtube.com/watch?v=NWOL8yXL6xI&list=PLmyoWnoyCKo8epWKGHAm4m_SyzoYhslk5&index=12
4. https://www.youtube.com/watch?v=K-j704F6F7Q&list=PLmyoWnoyCKo8epWKGHAm4m_SyzoYhslk5&index=13
5. https://vision.middlebury.edu/stereo/data/scenes2021/
