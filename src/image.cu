#include "../include/image.cuh"
#include "../include/common.h"
#include "../include/functions.cuh"
#include "../libs/stb/stb_image.h"
#include "../libs/stb/stb_image_write.h"
#include <cuda_runtime.h>
#include <stdexcept>

Image::Image(const char *filename) {
    _filename = filename;

    int w, h, c;
    unsigned char *data = stbi_load(_filename, &w, &h, &c, 3);

    if (!data) {
        return;
    }

    _width = w;
    _height = h;
    _channels = c;

    _nBytes = w * h * c * sizeof(unsigned char);

    // Allocate space for the host copy.
    _h_data = (unsigned char *) malloc(_nBytes);
    for (int i = 0; i < w * h * c; i++) {
        _h_data[i] = data[i];
    }

    // Allocate space for the cuda copy.
    cudaMalloc((unsigned char **) &_d_data, _nBytes);

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
    _h_data = (unsigned char *) malloc(_nBytes);
    for (int i = 0; i < _width * _height * _channels; i++) {
        _h_data[i] = obj._h_data[i];
    }

    // Allocate space for the cuda copy.
    cudaMalloc((unsigned char **) &_d_data, _nBytes);
    cudaMemcpy(_d_data, obj._d_data, _nBytes, cudaMemcpyDeviceToDevice);
}

Image::~Image(void) {
    free(_h_data);
    cudaFree(_d_data);
}

Image Image::operator-(const Image &obj) {
    // Return if images have different sizes.
    if (_width != obj._width or _height != obj._height or _channels != obj._channels) {
        throw std::invalid_argument("images have different sizes");
    }

    Image result(obj);
    result.setDevice(_device);

    if (strcmp(_device, _validDevices[0]) == 0) {
        differenceOnHost(result.getData(), getData(), getWidth(), getHeight(), getChannels());
    } else {
        int blockSize = 1024;
        dim3 threads(blockSize, 1);
        dim3 blocks((getSize() + threads.x - 1) / threads.x, 1);
        differenceOnDevice<<<blocks, threads>>>(result.getData(), getData(), getWidth(), getHeight(), getChannels());
    }

    return result;
}

int Image::getChannels() {
    return _channels;
}

unsigned char *Image::getData() {
    if (strcmp(_device, _validDevices[0]) == 0) {
        return _h_data;
    } else {
        return _d_data;
    }
}

const char *Image::getDevice() {
    return _device;
}

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

const char *Image::getFilename() {
    return _filename;
}

int Image::getHeight() {
    return _height;
}

int Image::getSize() {
    return _width * _height * _channels;
}

int Image::getWidth() {
    return _width;
}

bool Image::isSynchronized() {
    unsigned char *h_d_data_copy = (unsigned char *) malloc(_nBytes);
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

    stbi_write_png(filename, _width, _height, _channels, _h_data, _width * _channels);
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

void Image::convolution(float *kernel, int kernelSide) {
    unsigned char *dataCopy;

    if (strcmp(_device, _validDevices[0]) == 0) {
        // Create a copy of the data on host.
        dataCopy = (unsigned char *) malloc(_nBytes);
        for (int i = 0; i < getSize(); i++) {
            dataCopy[i] = getData()[i];
        }

        convolutionOnHost(getData(), dataCopy, kernel, kernelSide, getWidth(), getHeight(),
                          getChannels());
    } else {
        // Create a copy of the data on device.
        cudaMalloc((unsigned char **) &dataCopy, _nBytes);
        cudaMemcpy(dataCopy, getData(), _nBytes, cudaMemcpyDeviceToDevice);

        // Copy kernel to device.
        float *d_kernel;
        cudaMalloc((float **) &d_kernel, kernelSide * kernelSide * sizeof(float));
        cudaMemcpy(d_kernel, kernel, kernelSide * kernelSide * sizeof(float), cudaMemcpyHostToDevice);

        int blockSize = 1024;
        dim3 threads(blockSize, 1);
        dim3 blocks((getWidth() * getHeight() + threads.x - 1) / threads.x, 1);
        convolutionOnDevice<<<blocks, threads>>>(getData(), dataCopy, d_kernel, kernelSide, getWidth(), getHeight(),
                                                 getChannels());

        // Free memory.
        cudaFree(dataCopy);
    }
}

unsigned char *Image::histogram() {
    size_t histBytes = PIXEL_VALUES * getChannels() * sizeof(unsigned char);
    unsigned char *histogram = (unsigned char *) malloc(histBytes);

    for (int i = 0; i < PIXEL_VALUES * getChannels(); i++) {
        histogram[i] = 0;
    }

    if (strcmp(_device, _validDevices[0]) == 0) {
        histogramOnHost(histogram, getData(), getWidth(), getHeight(), getChannels());
    } else {
        // Copy histogram to device.
        unsigned char *d_histogram;
        cudaMalloc((unsigned char **) &d_histogram, histBytes);
        cudaMemcpy(d_histogram, histogram, histBytes, cudaMemcpyHostToDevice);

        int blockSize = 1024;
        dim3 threads(blockSize, 1);
        dim3 blocks((getSize() + threads.x - 1) / threads.x, 1);
        histogramOnDevice<<<blocks, threads>>>(d_histogram, getData(), getWidth(), getHeight(), getChannels());

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
        dataCopy = (unsigned char *) malloc(_nBytes);
        for (int i = 0; i < getSize(); i++) {
            dataCopy[i] = getData()[i];
        }

        rotateOnHost(getData(), dataCopy, rad, getWidth(), getHeight(), getChannels());
    } else {
        // Copy histogram to device.
        cudaMalloc((unsigned char **) &dataCopy, _nBytes);
        cudaMemcpy(dataCopy, _d_data, _nBytes, cudaMemcpyDeviceToDevice);

        int blockSize = 1024;
        dim3 threads(blockSize, 1);
        dim3 blocks((getWidth() * getHeight() + threads.x - 1) / threads.x, 1);
        rotateOnDevice<<<blocks, threads>>>(getData(), dataCopy, rad, getWidth(), getHeight(), getChannels());
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
        transposeOnDevice<<<blocks, threads>>>(getData(), getWidth(), getHeight(), getChannels());
    }
}
