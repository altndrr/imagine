#ifndef IMAGINE_IMAGE_H
#define IMAGINE_IMAGE_H

#define STB_IMAGE_IMPLEMENTATION
#define STB_IMAGE_WRITE_IMPLEMENTATION

#include <stdlib.h>

class Image {
public:
    Image(const char *filename);

    void dispose();

    int getChannels();

    unsigned char *getData();

    const char *getDevice();

    unsigned int *getElement(int index);

    unsigned int *getElement(int row, int col);

    const char *getFilename();

    int getHeight();

    int getSize();

    int getWidth();

    bool isSynchronized();

    void save(const char *filename);

    void setDevice(const char *device);


private:
    const char *_validDevices[2] = {"cpu", "cuda"};

    const char *_device = "cpu";
    const char *_filename;
    int _width, _height, _channels;
    unsigned char *_d_data, *_h_data;
    size_t _nBytes;
};

#endif //IMAGINE_IMAGE_H
