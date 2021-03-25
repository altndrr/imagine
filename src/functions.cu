#include "../include/functions.cuh"
#include "../include/common.h"

void convolutionOnHost(unsigned char *dst, unsigned char *src, float *kernel, int kernelSide,
                       const int width, const int height, const int channels) {
    unsigned int margin = int((kernelSide - 1) / 2);

    // Loop through each pixel.
    for (int y = margin; y < height - margin; y++) {
        for (int x = margin; x < width - margin; x++) {

            // Loop through each element of the kernel.
            for (int dy = 0; dy < kernelSide; dy++) {
                for (int dx = 0; dx < kernelSide; dx++) {

                    // Loop through the channels of the image.
                    for (int c = 0; c < channels; c++) {
                        int src_i = channels * ((y + (dy - margin)) * width + (x + (dx - margin))) + c;
                        int ker_i = dy * kernelSide + dx;
                        int dst_i = channels * (y * width + x) + c;

                        // Reset dst element at the start of the conv.
                        if (ker_i == 0) {
                            dst[dst_i] = 0;
                        }

                        // Add result of multiplication.
                        dst[dst_i] += int(src[src_i] * kernel[ker_i]);
                    }
                }
            }
        }
    }
}

__global__ void convolutionOnDevice(unsigned char *dst, unsigned char *src, float *kernel, int kernelSide,
                                    const int width, const int height, const int channels) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    // Check for overflow.
    if (i >= width * height) {
        return;
    }

    unsigned int margin = int((kernelSide - 1) / 2);

    int x = (i % width);
    int y = (int) i / width;

    // Check for minimum padding.
    if (y < margin or y > height - margin - 1 or x < margin or x > width - margin - 1) {
        return;
    }

    // Loop through each element of the kernel.
    for (int dy = 0; dy < kernelSide; dy++) {
        for (int dx = 0; dx < kernelSide; dx++) {

            // Loop through the channels of the image.
            for (int c = 0; c < channels; c++) {
                int src_i = channels * ((y + (dy - margin)) * width + (x + (dx - margin))) + c;
                int ker_i = dy * kernelSide + dx;
                int dst_i = channels * i + c;

                // Reset dst element at the start of the conv.
                if (ker_i == 0) {
                    dst[dst_i] = 0;
                }

                // Add result of multiplication.
                dst[dst_i] += int(src[src_i] * kernel[ker_i]);
            }
        }
    }
}

void differenceOnHost(unsigned char *src, unsigned char *dst, const int width, const int height, const int channels) {
    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
            for (int c = 0; c < channels; c++) {
                int i = channels * (y * width + x) + c;
                if (dst[i] > src[i]) {
                    dst[i] = dst[i] - src[i];
                } else {
                    dst[i] = src[i] - dst[i];
                }
            }
        }
    }
}

__global__ void differenceOnDevice(unsigned char *src, unsigned char *dst, const int width, const int height,
                                   const int channels) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    // Check for overflow.
    if (i >= width * height * channels) {
        return;
    }

    if (dst[i] > src[i]) {
        dst[i] = dst[i] - src[i];
    } else {
        dst[i] = src[i] - dst[i];
    }
}

void histogramOnHost(unsigned char *src, unsigned char *dst, const int width, const int height, const int channels) {
    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
            for (int c = 0; c < channels; c++) {
                int ia = channels * (y * width + x) + c;
                int ib = PIXEL_VALUES * c + int(src[ia]);

                dst[ib] += 1;
            }
        }
    }
}

__global__ void histogramOnDevice(unsigned char *src, unsigned char *dst, const int width, const int height,
                                  const int channels) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    // Check for overflow.
    if (i > PIXEL_VALUES * channels) {
        return;
    }

    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
            for (int c = 0; c < channels; c++) {
                int ia = channels * (y * width + x) + c;
                int ib = PIXEL_VALUES * c + int(src[ia]);

                if (ib == i) {
                    dst[i] += 1;
                }
            }
        }
    }
}

void transposeOnHost(unsigned char *data, const int width, const int height, const int channels) {
    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
            for (int c = 0; c < channels; c++) {
                int ia = channels * (y * width + x) + c;
                int ib = channels * (x * height + y) + c;

                if (ia > ib) {
                    continue;
                }

                unsigned char temp = data[ib];
                data[ib] = data[ia];
                data[ia] = temp;
            }
        }
    }
}


__global__ void transposeOnDevice(unsigned char *data, const int width, const int height, const int channels) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    // Check for overflow.
    if (i >= width * height) {
        return;
    }

    for (int c = 0; c < channels; c++) {
        int ia = channels * i + c;
        int ib = channels * ((i % width) * height + ((int) i / width)) + c;

        if (ia > ib) {
            continue;
        }

        unsigned char temp = data[ib];
        data[ib] = data[ia];
        data[ia] = temp;
    }
}