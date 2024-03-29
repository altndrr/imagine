#ifndef IMAGINE_IMAGE_H
#define IMAGINE_IMAGE_H

#define STB_IMAGE_IMPLEMENTATION
#define STB_IMAGE_WRITE_IMPLEMENTATION

#include <stdlib.h>

class Image {
public:
    Image(const char *filename, bool grayscale = false);

    Image(const Image &obj);

    ~Image(void);

    Image operator-(const Image &obj);

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

    void calcOpticalFlow(int *currentCorners, Image *previousFrame,
                         int *corners, int maxCorners, int levels = 2);

    void convolution(float *kernel, int kernelSide);

    void drawLine(int index1, int index2, int radius, int *color,
                  int colorSize);

    void drawLine(int x1, int y1, int x2, int y2, int radius, int *color,
                  int colorSize);

    void drawPoint(int index, int radius, int *color, int colorSize);

    void drawPoint(int x, int y, int radius, int *color, int colorSize);

    void findHomography(float *A, int *currentCorners, int *previousCorners,
                        int maxCorners);

    void goodFeaturesToTrack(int *corners, int maxCorners, float qualityLevel,
                             float minDistance);

    void rotate(double degree);

    void scale(float ratio);

    void translate(int px, int py);

    void transpose();

private:
    const char *_validDevices[2] = {"cpu", "cuda"};

    const char *_device = "cpu";
    const char *_filename;
    int _width, _height, _channels;
    unsigned char *_d_data, *_h_data;
    size_t _nBytes;
};

#endif // IMAGINE_IMAGE_H
