#ifndef IMAGINE_FUNCTIONS_Hconv
#define IMAGINE_FUNCTIONS_H

#include <cuda_runtime.h>

void convolutionOnHost(unsigned char *dst, unsigned char *src, float *kernel, int kernelSide,
                       const int width, const int height, const int channels);

__global__ void convolutionOnDevice(unsigned char *dst, unsigned char *src, float *kernel, int kernelSide,
                                    const int width, const int height, const int channels);

void drawPointOnHost(unsigned char *data, int x, int y, int radius, int *color, int colorSize, int width, int height,
                     int channels);

__global__ void drawPointOnDevice(unsigned char *data, int x, int y, int radius, int *color, int colorSize, int width,
                                  int height, int channels);

void differenceOnHost(unsigned char *dst, unsigned char *src, const int width, const int height, const int channels);

__global__ void differenceOnDevice(unsigned char *dst, unsigned char *src, const int width, const int height,
                                   const int channels);

void goodFeaturesToTrackOnHost(unsigned char *gradX, unsigned char *gradY, int *corners, int maxCorners,
                               float qualityLevel, float minDistance, int width, int height);

void histogramOnHost(unsigned char *dst, unsigned char *src, const int width, const int height, const int channels);

__global__ void histogramOnDevice(unsigned char *dst, unsigned char *src, const int width, const int height,
                                  const int channels);

void rotateOnHost(unsigned char *dst, unsigned char *src, const double radian, const int width, const int height,
                  const int channels);

__global__ void rotateOnDevice(unsigned char *dst, unsigned char *src, const double radian, const int width,
                               const int height, const int channels);

void transposeOnHost(unsigned char *data, const int width, const int height, const int channels);

__global__ void transposeOnDevice(unsigned char *data, const int width, const int height, const int channels);

void sumOfMatmulOnHost(float *total, float *A, float *B, int side);

#endif //IMAGINE_FUNCTIONS_H
