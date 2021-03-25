#ifndef IMAGINE_FUNCTIONS_Hconv
#define IMAGINE_FUNCTIONS_H

#include <cuda_runtime.h>

void convolutionOnHost(unsigned char *dst, unsigned char *src, float *kernel, int kernelSide,
                       const int width, const int height, const int channels);

__global__ void convolutionOnDevice(unsigned char *dst, unsigned char *src, float *kernel, int kernelSide,
                                    const int width, const int height, const int channels);

void histogramOnHost(unsigned char *src, unsigned char *dst, const int width, const int height, const int channels);

__global__ void histogramOnDevice(unsigned char *src, unsigned char *dst, const int width, const int height,
                                   const int channels);

void transposeOnHost(unsigned char *data, const int width, const int height, const int channels);

__global__ void transposeOnDevice(unsigned char *data, const int width, const int height, const int channels);


#endif //IMAGINE_FUNCTIONS_H
