#include "../include/common.h"
#include "../include/functions.cuh"
#include "../include/image.cuh"
#include "../include/kernel.h"
#include "../libs/stb/stb_image.h"
#include "../libs/stb/stb_image_write.h"
#include <cuda_runtime.h>
#include <queue>
#include <stdexcept>

Image::Image(const char *filename, bool grayscale) {
    _filename = filename;

    int desiredChannels = 3;
    if (grayscale) {
        desiredChannels = 1;
    }

    int w, h, c;
    unsigned char *data = stbi_load(_filename, &w, &h, &c, desiredChannels);

    if (!data) {
        return;
    }

    _width = w;
    _height = h;
    _channels = desiredChannels;

    _nBytes = w * h * desiredChannels * sizeof(unsigned char);

    // Allocate space for the host copy.
    _h_data = (unsigned char *)malloc(_nBytes);
    for (int i = 0; i < w * h * desiredChannels; i++) {
        _h_data[i] = data[i];
    }

    // Allocate space for the cuda copy.
    cudaMalloc((unsigned char **)&_d_data, _nBytes);

    stbi_image_free(data);
}

Image::Image(const Image &obj) {
    _device = obj._device;
    _filename = obj._filename;
    _width = obj._width;
    _height = obj._height;
    _channels = obj._channels;
    _nBytes = _width * _height * _channels * sizeof(unsigned char);

    // Allocate space for the host copy.
    _h_data = (unsigned char *)malloc(_nBytes);
    for (int i = 0; i < _width * _height * _channels; i++) {
        _h_data[i] = obj._h_data[i];
    }

    // Allocate space for the cuda copy.
    cudaMalloc((unsigned char **)&_d_data, _nBytes);
    cudaMemcpy(_d_data, obj._d_data, _nBytes, cudaMemcpyDeviceToDevice);
}

Image::~Image(void) {
    free(_h_data);
    cudaFree(_d_data);
}

Image Image::operator-(const Image &obj) {
    // Return if images have different sizes.
    if (_width != obj._width or _height != obj._height or
        _channels != obj._channels) {
        throw std::invalid_argument("images have different sizes");
    }

    Image result(obj);
    result.setDevice(_device);

    if (strcmp(_device, _validDevices[0]) == 0) {
        differenceOnHost(result.getData(), getData(), getWidth(), getHeight(),
                         getChannels());
    } else {
        int blockSize = 1024;
        dim3 threads(blockSize, 1);
        dim3 blocks((getSize() + threads.x - 1) / threads.x, 1);
        differenceOnDevice<<<blocks, threads>>>(result.getData(), getData(),
                                                getWidth(), getHeight(),
                                                getChannels());
    }

    return result;
}

int Image::getChannels() { return _channels; }

unsigned char *Image::getData() {
    if (strcmp(_device, _validDevices[0]) == 0) {
        return _h_data;
    } else {
        return _d_data;
    }
}

const char *Image::getDevice() { return _device; }

unsigned int *Image::getElement(int index) {
    unsigned int *values = new unsigned int[_channels];

    if (index >= _width * _height) {
        return NULL;
    }

    // Synchronize matrices if needed.
    // TODO optimise synchronization to avoid to check everytime when on cuda.
    if (strcmp(_device, _validDevices[1]) == 0 and isSynchronized() == 0) {
        cudaMemcpy(_h_data, _d_data, _nBytes, cudaMemcpyDeviceToHost);
    }

    for (int c = 0; c < _channels; c++) {
        values[c] = _h_data[index * _channels + c];
    }

    return values;
}

unsigned int *Image::getElement(int row, int col) {
    if (row >= _height or col >= _width) {
        return NULL;
    }

    return getElement(row * _width + col);
}

const char *Image::getFilename() { return _filename; }

int Image::getHeight() { return _height; }

int Image::getSize() { return _width * _height * _channels; }

int Image::getWidth() { return _width; }

bool Image::isSynchronized() {
    unsigned char *h_d_data_copy = (unsigned char *)malloc(_nBytes);
    cudaMemcpy(h_d_data_copy, _d_data, _nBytes, cudaMemcpyDeviceToHost);

    float epsilon = 1.0E-8;
    int match = 1;
    for (int i = 0; i < getSize(); i++) {
        if (abs(_h_data[i] - h_d_data_copy[i]) > epsilon) {
            match = 0;
            break;
        }
    }

    free(h_d_data_copy);
    return match;
}

void Image::save(const char *filename) {
    // Synchronize matrices if needed.
    if (strcmp(_device, _validDevices[1]) == 0 and isSynchronized() == 0) {
        cudaMemcpy(_h_data, _d_data, _nBytes, cudaMemcpyDeviceToHost);
    }

    stbi_write_png(filename, _width, _height, _channels, _h_data,
                   _width * _channels);
}

void Image::setDevice(const char *device) {
    if (arrayContains(_validDevices, device) == 0) {
        return;
    }

    if (strcmp(device, _device) != 0) {
        _device = device;

        if (strcmp(device, _validDevices[0]) == 0) {
            cudaMemcpy(_h_data, _d_data, _nBytes, cudaMemcpyDeviceToHost);
        } else {
            cudaMemcpy(_d_data, _h_data, _nBytes, cudaMemcpyHostToDevice);
        }
    }
}

void Image::calcOpticalFlow(int *currentCorners, Image *previousFrame,
                            int *corners, int maxCorners, int levels) {
    Image gray(getFilename(), true);
    Image prevGray(previousFrame->getFilename(), true);
    gray.setDevice(getDevice());
    prevGray.setDevice(getDevice());

    if (strcmp(_device, _validDevices[0]) == 0) {
        unsigned char *currPyramidalScales[levels];
        unsigned char *prevPyramidalScales[levels];

        // Create the pyramidal scales.
        for (int l = 0; l < levels; l++) {
            int levelWidth = gray.getWidth() / pow(2, l);
            int levelHeight = gray.getHeight() / pow(2, l);
            currPyramidalScales[l] = new unsigned char[gray.getSize()];
            prevPyramidalScales[l] = new unsigned char[prevGray.getSize()];

            if (l == 0) {
                for (int i = 0; i < gray.getSize(); i++) {
                    currPyramidalScales[l][i] = gray.getData()[i];
                    prevPyramidalScales[l][i] = prevGray.getData()[i];
                }
            } else {
                scaleOnHost(currPyramidalScales[l], currPyramidalScales[l - 1],
                            0.5, levelWidth * 2, levelHeight * 2, 1);
                scaleOnHost(prevPyramidalScales[l], prevPyramidalScales[l - 1],
                            0.5, levelWidth * 2, levelHeight * 2, 1);
            }
        }

        opticalFLowOnHost(currentCorners, corners, maxCorners,
                          currPyramidalScales, prevPyramidalScales, levels,
                          gray.getWidth(), gray.getHeight());
    } else {
        // Copy corner arrays to device.
        size_t cornersBytes = maxCorners * sizeof(int);
        int *d_corners, *d_currCorners;
        cudaMalloc((int **)&d_corners, cornersBytes);
        cudaMalloc((int **)&d_currCorners, cornersBytes);
        cudaMemcpy(d_corners, corners, cornersBytes, cudaMemcpyHostToDevice);

        // Create the pyramidal scales.
        size_t pyramidBytes = gray.getSize() * sizeof(unsigned char);
        unsigned char *currPyramidalScales, *prevPyramidalScales;
        cudaMalloc((unsigned char **)&currPyramidalScales,
                   levels * pyramidBytes);
        cudaMalloc((unsigned char **)&prevPyramidalScales,
                   levels * pyramidBytes);
        int copyBlockSize = 1024;
        dim3 copyThreads(copyBlockSize, 1);
        dim3 copyBlocks(
            (gray.getWidth() * gray.getHeight() + copyThreads.x - 1) /
                copyThreads.x,
            1);

        for (int l = 0; l < levels; l++) {
            int levelWidth = gray.getWidth() / pow(2, l);
            int levelHeight = gray.getHeight() / pow(2, l);

            if (l == 0) {
                cudaMemcpy(currPyramidalScales, gray.getData(), pyramidBytes,
                           cudaMemcpyDeviceToDevice);
                cudaMemcpy(prevPyramidalScales, prevGray.getData(),
                           pyramidBytes, cudaMemcpyDeviceToDevice);
            } else {
                scaleOnDevice<<<copyBlocks, copyThreads>>>(
                    currPyramidalScales + l * gray.getSize(),
                    currPyramidalScales + (l - 1) * gray.getSize(), 0.5,
                    levelWidth * 2, levelHeight * 2, 1);
                scaleOnDevice<<<copyBlocks, copyThreads>>>(
                    prevPyramidalScales + l * gray.getSize(),
                    prevPyramidalScales + (l - 1) * gray.getSize(), 0.5,
                    levelWidth * 2, levelHeight * 2, 1);
            }
        }

        int blockSize = 1024;
        dim3 threads(blockSize, 1);
        dim3 blocks((maxCorners + copyThreads.x - 1) / copyThreads.x, 1);
        opticalFLowOnDevice<<<blocks, threads>>>(
            d_currCorners, d_corners, maxCorners, currPyramidalScales,
            prevPyramidalScales, levels, gray.getSize(), gray.getWidth(),
            gray.getHeight());

        cudaMemcpy(currentCorners, d_currCorners, cornersBytes,
                   cudaMemcpyDeviceToHost);

        // Free memory.
        cudaFree(d_corners);
        cudaFree(d_currCorners);
        cudaFree(currPyramidalScales);
        cudaFree(prevPyramidalScales);
    }
}

void Image::convolution(float *kernel, int kernelSide) {
    unsigned char *dataCopy;

    if (strcmp(_device, _validDevices[0]) == 0) {
        // Create a copy of the data on host.
        dataCopy = (unsigned char *)malloc(_nBytes);
        for (int i = 0; i < getSize(); i++) {
            dataCopy[i] = getData()[i];
        }

        convolutionOnHost(getData(), dataCopy, kernel, kernelSide, getWidth(),
                          getHeight(), getChannels());
    } else {
        // Create a copy of the data on device.
        cudaMalloc((unsigned char **)&dataCopy, _nBytes);
        cudaMemcpy(dataCopy, getData(), _nBytes, cudaMemcpyDeviceToDevice);

        // Copy kernel to device.
        float *d_kernel;
        cudaMalloc((float **)&d_kernel,
                   kernelSide * kernelSide * sizeof(float));
        cudaMemcpy(d_kernel, kernel, kernelSide * kernelSide * sizeof(float),
                   cudaMemcpyHostToDevice);

        int blockSize = 1024;
        dim3 threads(blockSize, 1);
        dim3 blocks((getWidth() * getHeight() + threads.x - 1) / threads.x, 1);
        convolutionOnDevice<<<blocks, threads>>>(getData(), dataCopy, d_kernel,
                                                 kernelSide, getWidth(),
                                                 getHeight(), getChannels());

        // Free memory.
        cudaFree(dataCopy);
    }
}

void Image::drawLine(int index1, int index2, int radius, int *color,
                     int colorSize) {
    if (index1 < 0 or index2 < 0) {
        return;
    }

    int x1 = (int)(index1 / getWidth());
    int y1 = (index1 % getWidth());
    int x2 = (int)(index2 / getWidth());
    int y2 = (index2 % getWidth());
    this->drawLine(x1, y1, x2, y2, radius, color, colorSize);
}

void Image::drawLine(int x1, int y1, int x2, int y2, int radius, int *color,
                     int colorSize) {
    if (strcmp(_device, _validDevices[0]) == 0) {
        drawLineOnHost(getData(), x1, y1, x2, y2, radius, color, colorSize,
                       getWidth(), getHeight(), getChannels());
    } else {
        int blockSize = 1024;
        dim3 threads(blockSize, 1);
        dim3 blocks((getWidth() * getHeight() + threads.x - 1) / threads.x, 1);

        size_t colorBytes = colorSize * sizeof(int);
        int *d_color;
        cudaMalloc((int **)&d_color, colorBytes);
        cudaMemcpy(d_color, color, colorBytes, cudaMemcpyHostToDevice);

        drawLineOnDevice<<<blocks, threads>>>(getData(), x1, y1, x2, y2, radius,
                                              d_color, colorSize, getWidth(),
                                              getHeight(), getChannels());
    }
}

void Image::drawPoint(int index, int radius, int *color, int colorSize) {
    if (index < 0) {
        return;
    }

    int x = (int)(index / getWidth());
    int y = (index % getWidth());
    this->drawPoint(x, y, radius, color, colorSize);
}

void Image::drawPoint(int x, int y, int radius, int *color, int colorSize) {
    if (strcmp(_device, _validDevices[0]) == 0) {
        drawPointOnHost(getData(), x, y, radius, color, colorSize, getWidth(),
                        getHeight(), getChannels());
    } else {
        int blockSize = 1024;
        dim3 threads(blockSize, 1);
        dim3 blocks((getWidth() * getHeight() + threads.x - 1) / threads.x, 1);

        size_t colorBytes = colorSize * sizeof(int);
        int *d_color;
        cudaMalloc((int **)&d_color, colorBytes);
        cudaMemcpy(d_color, color, colorBytes, cudaMemcpyHostToDevice);

        drawPointOnDevice<<<blocks, threads>>>(getData(), x, y, radius, d_color,
                                               colorSize, getWidth(),
                                               getHeight(), getChannels());
    }
}

void Image::findHomography(float *A, int *currentCorners, int *previousCorners,
                           int maxCorners) {
    const int maxIter = 2000;
    const int N_POINTS = 3;
    const int SPACE_DIM = 2;

    if (strcmp(_device, _validDevices[0]) == 0) {
        // Estimate maxIter different rigid transformations.
        // The algorithm estimates a matrix using a triplet of points.
        float *matrices = new float[N_POINTS * (SPACE_DIM + 1) * maxIter];
        float *scores = new float[maxIter];
        findHomographyRANSACOnHost(matrices, scores, maxIter, currentCorners,
                                   previousCorners, maxCorners, getWidth(),
                                   getHeight());

        int bestMatrix = -1;
        float minError = INFINITY;

        for (int i = 0; i < maxIter; i++) {
            if (scores[i] < minError) {
                bestMatrix = i;
                minError = scores[i];
            }
        }

        int offset = bestMatrix * (N_POINTS * (SPACE_DIM + 1));
        for (int i = 0; i < N_POINTS * (SPACE_DIM + 1); i++) {
            if (minError < INFINITY) {
                A[i] = matrices[offset + i];
            } else {
                // If the minError is INFINITY, then set the transformation to
                // the identity matrix to avoid any type of transformation.
                int side = sqrt(N_POINTS * (SPACE_DIM + 1));
                A[i] = int(i % side == (int)i / side);
            }
        }

        delete[] matrices;
        delete[] scores;
    }
}

void Image::goodFeaturesToTrack(int *corners, int maxCorners,
                                float qualityLevel, float minDistance) {
    Image gradX(getFilename(), true);
    Image gradY(getFilename(), true);
    gradX.setDevice(getDevice());
    gradY.setDevice(getDevice());

    int side;
    float *sobelX, *sobelY;
    Kernel::SobelX(&sobelX, &side);
    Kernel::SobelY(&sobelY);
    gradX.convolution(sobelX, side);
    gradY.convolution(sobelY, side);

    int scoreSize = getWidth() * getHeight();
    float *scoreMatrix = new float[scoreSize];

    if (strcmp(_device, _validDevices[0]) == 0) {
        cornerScoreOnHost(gradX.getData(), gradY.getData(), scoreMatrix,
                          getWidth(), getHeight());
    } else {
        // Copy corner array to device.
        size_t scoreMatrixBytes = scoreSize * sizeof(float);
        float *d_scoreMatrix;
        cudaMalloc((float **)&d_scoreMatrix, scoreMatrixBytes);
        cudaMemcpy(d_scoreMatrix, scoreMatrix, scoreMatrixBytes,
                   cudaMemcpyHostToDevice);

        int blockSize = 1024;
        dim3 threads(blockSize, 1);
        dim3 blocks((scoreSize + threads.x - 1) / threads.x, 1);
        cornerScoreOnDevice<<<blocks, threads>>>(gradX.getData(),
                                                 gradY.getData(), d_scoreMatrix,
                                                 getWidth(), getHeight());

        // Copy result to host.
        cudaMemcpy(scoreMatrix, d_scoreMatrix, scoreMatrixBytes,
                   cudaMemcpyDeviceToHost);
        cudaFree(d_scoreMatrix);
    }

    // Create a priority queue of the scores and store the highest score.
    std::priority_queue<std::pair<float, int>> qR;
    float strongestScore = 0.00;
    for (int i = 0; i < scoreSize; i++) {
        // Skip nan values.
        if (scoreMatrix[i] != scoreMatrix[i]) {
            continue;
        }

        qR.push(std::pair<float, int>(scoreMatrix[i], i));
        if (strongestScore < scoreMatrix[i]) {
            strongestScore = scoreMatrix[i];
        }
    }

    // Extract the top-K corners.
    float threshold = strongestScore * qualityLevel;
    for (int i = 0; i < maxCorners; ++i) {
        corners[i] = -1;
        float kValue;
        int kIndex;
        bool isDistant;

        do {
            kValue = qR.top().first;
            kIndex = qR.top().second;
            isDistant = true;

            // Evaluate the Euclidean distance to the previous corners.
            int j = 0;
            while (j < i and isDistant) {
                int otherIndex = corners[j];
                int dx =
                    ((int)otherIndex / getWidth()) - ((int)kIndex / getWidth());
                int dy = (otherIndex % getWidth()) - (kIndex % getWidth());
                int dist = sqrt(pow(dx, 2) + pow(dy, 2));

                isDistant = dist > minDistance;
                j++;
            }

            if (isDistant) {
                // Add only if score is high enough.
                if (kValue >= threshold) {
                    corners[i] = kIndex;
                }
            }

            qR.pop();
        } while (not isDistant);
    }
}

unsigned char *Image::histogram() {
    size_t histBytes = PIXEL_VALUES * getChannels() * sizeof(unsigned char);
    unsigned char *histogram = (unsigned char *)malloc(histBytes);

    for (int i = 0; i < PIXEL_VALUES * getChannels(); i++) {
        histogram[i] = 0;
    }

    if (strcmp(_device, _validDevices[0]) == 0) {
        histogramOnHost(histogram, getData(), getWidth(), getHeight(),
                        getChannels());
    } else {
        // Copy histogram to device.
        unsigned char *d_histogram;
        cudaMalloc((unsigned char **)&d_histogram, histBytes);
        cudaMemcpy(d_histogram, histogram, histBytes, cudaMemcpyHostToDevice);

        int blockSize = 1024;
        dim3 threads(blockSize, 1);
        dim3 blocks((getSize() + threads.x - 1) / threads.x, 1);
        histogramOnDevice<<<blocks, threads>>>(
            d_histogram, getData(), getWidth(), getHeight(), getChannels());

        // Copy result to host.
        cudaMemcpy(histogram, d_histogram, histBytes, cudaMemcpyDeviceToHost);
        cudaFree(d_histogram);
    }

    return histogram;
}

void Image::rotate(double degree) {
    double rad = degree * (M_PI / 180);
    unsigned char *dataCopy;

    if (strcmp(_device, _validDevices[0]) == 0) {
        // Create a copy of the data on host.
        dataCopy = (unsigned char *)malloc(_nBytes);
        for (int i = 0; i < getSize(); i++) {
            dataCopy[i] = getData()[i];
        }

        rotateOnHost(getData(), dataCopy, rad, getWidth(), getHeight(),
                     getChannels());
    } else {
        // Copy histogram to device.
        cudaMalloc((unsigned char **)&dataCopy, _nBytes);
        cudaMemcpy(dataCopy, _d_data, _nBytes, cudaMemcpyDeviceToDevice);

        int blockSize = 1024;
        dim3 threads(blockSize, 1);
        dim3 blocks((getWidth() * getHeight() + threads.x - 1) / threads.x, 1);
        rotateOnDevice<<<blocks, threads>>>(
            getData(), dataCopy, rad, getWidth(), getHeight(), getChannels());
    }
}

void Image::scale(float ratio) {
    // Return if ratio is invalid.
    if (ratio == 1.0 or ratio < 0.0) {
        return;
    }

    unsigned char *newData;
    int newWidth = int(getWidth() * ratio);
    int newHeight = int(getHeight() * ratio);
    int newBytes = newWidth * newHeight * getChannels() * sizeof(unsigned char);

    if (strcmp(_device, _validDevices[0]) == 0) {
        newData = (unsigned char *)malloc(newBytes);
        scaleOnHost(newData, getData(), ratio, getWidth(), getHeight(),
                    getChannels());

        // Update data both on device and on host.
        free(_h_data);
        _h_data = newData;
        cudaFree(_d_data);
        cudaMalloc((unsigned char **)&_d_data, newBytes);
        cudaMemcpy(_d_data, _h_data, newBytes, cudaMemcpyHostToDevice);
    } else {
        int blockSize = 1024;
        dim3 threads(blockSize, 1);
        dim3 blocks((newWidth * newHeight + threads.x - 1) / threads.x, 1);

        cudaMalloc((unsigned char **)&newData, newBytes);
        scaleOnDevice<<<blocks, threads>>>(
            newData, getData(), ratio, getWidth(), getHeight(), getChannels());

        // Update data both on device and on host.
        cudaFree(_d_data);
        cudaMalloc((unsigned char **)&_d_data, newBytes);
        cudaMemcpy(_d_data, newData, newBytes, cudaMemcpyDeviceToDevice);
        free(_h_data);
        _h_data = (unsigned char *)malloc(newBytes);
        cudaMemcpy(_h_data, _d_data, newBytes, cudaMemcpyDeviceToHost);
    }

    // Update other attributes.
    _width = newWidth;
    _height = newHeight;
    _nBytes = newBytes;
}

void Image::translate(int px, int py) {
    unsigned char *dataCopy;

    if (strcmp(_device, _validDevices[0]) == 0) {
        // Create a copy of the data on host.
        dataCopy = (unsigned char *)malloc(_nBytes);
        for (int i = 0; i < getSize(); i++) {
            dataCopy[i] = getData()[i];
        }

        translateOnHost(getData(), dataCopy, px, py, getWidth(), getHeight(),
                        getChannels());
    } else {
        // Copy histogram to device.
        cudaMalloc((unsigned char **)&dataCopy, _nBytes);
        cudaMemcpy(dataCopy, _d_data, _nBytes, cudaMemcpyDeviceToDevice);

        int blockSize = 1024;
        dim3 threads(blockSize, 1);
        dim3 blocks((getWidth() * getHeight() + threads.x - 1) / threads.x, 1);
        translateOnDevice<<<blocks, threads>>>(getData(), dataCopy, px, py,
                                               getWidth(), getHeight(),
                                               getChannels());
    }
}

void Image::transpose() {
    // Return if width and height are different.
    if (getWidth() != getHeight()) {
        throw std::invalid_argument("width and height must have the same size");
    }

    if (strcmp(_device, _validDevices[0]) == 0) {
        transposeOnHost(getData(), getWidth(), getHeight(), getChannels());
    } else {
        int blockSize = 1024;
        dim3 threads(blockSize, 1);
        dim3 blocks((getWidth() * getHeight() + threads.x - 1) / threads.x, 1);
        transposeOnDevice<<<blocks, threads>>>(getData(), getWidth(),
                                               getHeight(), getChannels());
    }
}
