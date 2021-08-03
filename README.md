# Imagine

GPU accelerated library for image processing.

## Purpose
The purpose of this project is to develop a suite of tools and algorithms for image processing.
The constraints of the development are to build a library with GPU acceleration and with the least number of external dependencies.
Aside from [stb](https://github.com/nothings/stb) to read and write image files, there are no other external imports.

## Setup
To initialise the environment, [CUDA Toolkit](https://developer.nvidia.com/cuda-toolkit) must be installed on the system.

### Clone the repository
```shell
git clone https://github.com/altndrr/imagine.git
```

### Compile the source code
The execution of the command below will create a `./imagine` file in the code directory.
```shell
make
```

## Usage
The repository exposes a command-line interface to present its capabilities.
```text
Usage:
  imagine convolution [--cuda | --gray | --time]
  imagine difference [--cuda | --gray | --time]
  imagine rotate [--cuda | --gray | --time]
  imagine scale [--cuda | --gray | --time]
  imagine translate [--cuda | --gray | --time]
  imagine transpose [--cuda | --gray | --time]
  imagine point [--cuda | --gray | --time]
  imagine line [--cuda | --gray | --time]
  imagine shi [--cuda | --gray | --time]
  imagine lucas [--cuda | --gray | --time]
  imagine homography [--cuda | --gray | --time]
  imagine stabilise [--cuda | --gray | --time]
  imagine (-h | --help)
  imagine --version

Options:
  --cuda        Execute code on GPU.
  --gray        Convert input to grayscale.
  --time        Print the execution time.
  -h --help     Show this screen.
  --version     Show version.
```

## Structure
The repository considers three main files.
 1. `main.cpp`: contains the CLI and the code to provide some examples of the functionalities.
 2. `image.cu`: defines an Image class that represents an image file. The user can move instances of the image class between CPU and cuda with the command `image.setDevice(device)`. This class also provides a wrapper around all the image processing functions, either on CPU or GPU.
 3. `functions.cu`: contains pairs of image processing functions (and other helper methods), both for CPU and GPU.

### Utils library
For videos, the repository contains a utils library written in Python.
The two primary operations provided are implosion and explosion.
The first one creates a video from a set of images, while the second one splits a video into frames.
The initialisation of the environment requires Python3, pip and venv.
Both pip and venv should come with the installation of Python.
If not, once can install them with:
```shell
sudo apt-get install python3-pip
pip install venv
```

To initialise the environment run (in the repository folder):
```shell
python -m venv utils/venv
source utils/venv/bin/activate
pip install -r requirements.txt
```

Below, it is possible to find the CLI of the utils module.
```shell
Usage:
    utils explode <video> [-n NUM]
    utils implode <folder> [--fps VAL]
    utils -h | --help

Options:
    --fps VAL                   Number of frame per second [default: 25].
    -n NUM --num=NUM            Number of frames.
    -o FILE --output=FILE       Path to save results.
    -h --help                   Show this screen.
```

## Workflow
One can use any of the operations found in this repository in whichever order one prefers.
The primary commands of the repository are below.
Their descriptions are available to explain their functionalities and show some usage examples.

### Basic operations
#### Convolution
Perform a convolution operation on an input image with a specified kernel.
The `kernel.cpp` file provides some basic kernels to use in place of user-defined ones.
The matrices available are Gaussian blur, Sobel X and Sobel Y.

```c++
float kernel[] = {0, -1, 0, -1, 4, -1, 0, -1, 0};
Image image("data/lena.png");
image.convolution(kernel, 3);
```

#### Difference
Evaluate the absolute difference between two input images.
The two images must share the same width and height.
The first element of the subtraction defines the device used for the operation.

```c++
Image image1("data/lena.png");
Image image2("data/output.png");
Image res = image1 - image2;
```
#### Rotate, scale, translate, transpose
Rotation takes as input the degrees to rotate the image.
The technique employed for this operation is Rotation by Area Mapping (RAM).
This implementation collects for each pixel the four source pixels that partially covers it.
Given these points, it then computes the value of the destination pixel as their area-weighted average.

Regarding scaling and translation, they require either the scaling factor or the two displacements (x and y).
Lastly, transposition inverts each point's coordinates and works only for square images.

```c++
Image image("data/lena.png");
image.rotate(90.0f);
image.scale(0.5);
image.translate(5, 5);
image.transpose();
```

### Drawing
#### Point
Given an input image, it draws a point on it.
The method considers the x and y position of the point alongside its radius and colour.

```c++
int colour[] = {255, 0, 0};
Image image("data/lena.png");
image.drawPoint(100, 100, 25, colour, 3);
```

#### Line
This operation draws a line on an input image.
It is almost identical to the previous method, but it requires another pair of x and y coordinates.

```c++
int colour[] = {255, 0, 0};
Image image("data/lena.png");
image.drawLine(10, 10, 70, 70, 5, colour, 3);
```

### Algorithms
#### Shi-Tomasi "Good Features to Track"
The Shi-Tomasi algorithm for corner detection [1] enables the search of "good features to track".
Given the number of corners, it selects the pixels associated with the highest variation in contrast in both the x and y directions.
This algorithm extends the Harris solution by simplifying the scoring function of corners.

The method considers two additional parameters (other than the corner array and its size).
Namely, one could define the minimum distance between two selected corners and their quality level.
The latter defines the minimum score a corner must have to be considered a good feature, and its value is relative to the best corner.

```c++
int maxCorners = 100;
int *corners = new int[maxCorners];
Image image("data/lena.png");
image.goodFeaturesToTrack(corners, maxCorners, 0.05, 5.0);

int color[] = {0, 255, 0};
for (int i = 0; i < maxCorners; i++) {
    image.drawPoint(corners[i], 2, color, 3);
}
```

#### Lucas-Kanade "Optical Flow"
The Lucas-Kanade optical flow [2] is an algorithm for tracking features points in an image.
It is based on optical flow estimation and assumes that it is essentially constant in a local neighbourhood.
The algorithm has a pyramidal implementation to guarantee both accuracy and robustness.
More precisely, it considers multiple downscaling of the same image and searches for the displacement of the feature points from the lowest to the highest level.

The last parameter of the method `calcOpticalFlow` defines the number of levels to consider for the pyramid.

```c++
int maxCorners = 100;
int *corners = new int[maxCorners];
Image image("data/lena.png");
image.goodFeaturesToTrack(corners, maxCorners, 0.05, 5.0);

int *currCorners = new int[maxCorners];
image2.calcOpticalFlow(currCorners, &image1, corners, maxCorners, 2);

int red[] = {255, 0, 0};
int green[] = {0, 255, 0};
for (int i = 0; i < maxCorners; i++) {
    image2.drawLine(corners[i], currCorners[i], 1, green, 3);
    if (corners[i] != currCorners[i])
        image2.drawPoint(currCorners[i], 1, red, 3);
}
```

#### Homography estimation
Given two images and the optical flow between them, it is possible to evaluate the transformation that minimises the reconstruction error.
More precisely, it would be possible to discover the matrix that transforms the first frame into the second one.

The algorithm uses Random sample consensus (RANSAC) [3] to estimate the parameters of the matrix.

```c++
int maxCorners = 100;
int *corners = new int[maxCorners];
image1.goodFeaturesToTrack(corners, maxCorners, 0.05, 5.0);

int *currCorners = new int[maxCorners];
image2.calcOpticalFlow(currCorners, &image1, corners, maxCorners, 2);

float *A = new float[9];
image2.findHomography(A, currCorners, corners, maxCorners);
float dx = A[2];
float dy = A[5];
float da = atan2(A[3], A[0]);
image2.rotate(da);
image2.translate(dx, dy);
```

#### Extra: stabilise
With all the above methods, it is possible to develop a video stabilisation algorithm that exploits the optical flow in the image to find the homography from the previous image to the current one.
Afterwards, one could use this information to evaluate the displacement and the rotation one needs to apply on the current frame to reduce the camera movements.
If applied to each frame, it would be possible to evaluate the moving average of the rotation and displacement on x and y.
With the consecutive application of these values to each frame, the resulting images would present more soft camera movements.

##### Usage example
```shell
source utils/venv/bin/actiate
python -m utils explode path/to/video
mv path/to/exploded_video data/inputs
./imagine stabilise --cuda --time
python -m utils implode data/outputs
```

## References
 1. Shi, Jianbo. "Good features to track." 1994 Proceedings of IEEE conference on computer vision and pattern recognition. IEEE, 1994.
 2. Lucas, Bruce D., and Takeo Kanade. "An iterative image registration technique with an application to stereo vision." 1981.
 3. Fischler, Martin A., and Robert C. Bolles. "Random sample consensus: a paradigm for model fitting with applications to image analysis and automated cartography." Communications of the ACM 24.6 (1981): 381-395.