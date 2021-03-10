#include "../include/image.h"
#include "../libs/stb/stb_image.h"
#include "../libs/stb/stb_image_write.h"

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
    _data = data;
}

void Image::dispose() {
    stbi_image_free(_data);
}

int Image::getChannels() {
    return _channels;
}

unsigned int *Image::getElement(int index) {
    unsigned int *values = new unsigned int[_channels];

    if (index >= _width * _height) {
        return NULL;
    }

    for (int c = 0; c < _channels; c++) {
        values[c] = _data[index * _channels + c];
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

void Image::save(const char *filename) {
    stbi_write_png(filename, _width, _height, _channels, _data, _width * _channels);
}
