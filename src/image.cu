#include "../include/image.cuh"
#include "../include/common.h"
#include "../libs/stb/stb_image.h"
#include "../libs/stb/stb_image_write.h"
#include <cuda_runtime.h>

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

void Image::dispose() {
    free(_h_data);
    cudaFree(_d_data);
}

int Image::getChannels() {
    return _channels;
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
    return _height * _width;
}

int Image::getWidth() {
    return _width;
}

bool Image::isSynchronized() {
    unsigned char *h_d_data_copy = (unsigned char *) malloc(_nBytes);
    cudaMemcpy(h_d_data_copy, _d_data, _nBytes, cudaMemcpyDeviceToHost);

    float epsilon = 1.0E-8;
    int match = 1;
    for (int i = 0; i < getSize() * getChannels(); i++) {
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
