#ifndef IMAGINE_FUNCTIONS_Hconv
#define IMAGINE_FUNCTIONS_H

#include <cuda_runtime.h>

void convolutionOnHost(unsigned char *dst, unsigned char *src, float *kernel,
                       int kernelSide, const int width, const int height,
                       const int channels);

__global__ void convolutionOnDevice(unsigned char *dst, unsigned char *src,
                                    float *kernel, int kernelSide,
                                    const int width, const int height,
                                    const int channels);

void drawLineOnHost(unsigned char *data, int x1, int y1, int x2, int y2,
                    int radius, int *color, int colorSize, int width,
                    int height, int channels);

__global__ void drawLineOnDevice(unsigned char *data, int x1, int y1, int x2,
                                 int y2, int radius, int *color, int colorSize,
                                 int width, int height, int channels);

void drawPointOnHost(unsigned char *data, int x, int y, int radius, int *color,
                     int colorSize, int width, int height, int channels);

__global__ void drawPointOnDevice(unsigned char *data, int x, int y, int radius,
                                  int *color, int colorSize, int width,
                                  int height, int channels);

void differenceOnHost(unsigned char *dst, unsigned char *src, const int width,
                      const int height, const int channels);

__global__ void differenceOnDevice(unsigned char *dst, unsigned char *src,
                                   const int width, const int height,
                                   const int channels);

void cornerScoreOnHost(unsigned char *gradX, unsigned char *gradY, float *R,
                       int width, int height);

__global__ void cornerScoreOnDevice(unsigned char *gradX, unsigned char *gradY,
                                    float *R, int width, int height);

void histogramOnHost(unsigned char *dst, unsigned char *src, const int width,
                     const int height, const int channels);

__global__ void histogramOnDevice(unsigned char *dst, unsigned char *src,
                                  const int width, const int height,
                                  const int channels);

void rotateOnHost(unsigned char *dst, unsigned char *src, const double radian,
                  const int width, const int height, const int channels);

__global__ void rotateOnDevice(unsigned char *dst, unsigned char *src,
                               const double radian, const int width,
                               const int height, const int channels);

void scaleOnHost(unsigned char *dst, unsigned char *src, const double ratio,
                 const int width, const int height, const int channels);

__global__ void scaleOnDevice(unsigned char *dst, unsigned char *src,
                              const double ratio, const int width,
                              const int height, const int channels);

void transposeOnHost(unsigned char *data, const int width, const int height,
                     const int channels);

__global__ void transposeOnDevice(unsigned char *data, const int width,
                                  const int height, const int channels);

void sumOfMatmulOnHost(float *total, float *A, float *B, int side);

__device__ void sumOfMatmulOnDevice(float *total, float *A, float *B, int side);

float sumOfSquareDifferencesOnHost(unsigned char *patch1, unsigned char *patch2,
                                   int patchSide);

__device__ float sumOfSquareDifferencesOnDevice(unsigned char *patch1,
                                                unsigned char *patch2,
                                                int patchSide);

void extractPatchOnHost(unsigned char *patch, unsigned char *data,
                        int centerIndex, int patchSide, int width, int height);

__device__ void extractPatchOnDevice(unsigned char *patch, unsigned char *data,
                                     int centerIndex, int patchSide, int width,
                                     int height);

#endif // IMAGINE_FUNCTIONS_H
