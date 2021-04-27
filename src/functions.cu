#include "../include/functions.cuh"
#include "../include/common.h"
#include <cmath>

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

    int x = (int) i / width;
    int y = (i % width);

    // Check for minimum padding.
    if (y < margin or y > height - margin - 1 or x < margin or x > width - margin - 1) {
        return;
    }

    // Loop through each element of the kernel.
    for (int dy = 0; dy < kernelSide; dy++) {
        for (int dx = 0; dx < kernelSide; dx++) {

            // Loop through the channels of the image.
            for (int c = 0; c < channels; c++) {
                int src_i = channels * ((x + (dx - margin)) * width + (y + (dy - margin))) + c;
                int ker_i = dx * kernelSide + dy;
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

void drawPointOnHost(unsigned char *data, int x, int y, int radius, int *color, int colorSize, int width, int height,
                     int channels) {
    for (int dy = max(0, y - radius); dy < y + radius; dy++) {
        for (int dx = max(0, x - radius); dx < x + radius; dx++) {
            int index = (dx * width + dy) * channels;

            for (int c = 0; c < min(channels, colorSize); c++) {
                data[index + c] = color[c];
            }
        }
    }
}

__global__ void drawPointOnDevice(unsigned char *data, int x, int y, int radius, int *color, int colorSize, int width,
                                  int height, int channels) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    // Check for overflow.
    if (i >= width * height) {
        return;
    }

    int dx = (int) i / width;
    int dy = (i % width);

    // Check for point boundaries.
    if (dy < y - radius or dy >= y + radius or dx < x - radius or dx >= x + radius) {
        return;
    }

    for (int c = 0; c < min(channels, colorSize); c++) {
        int index = channels * i;
        data[index + c] = color[c];
    }
}

void differenceOnHost(unsigned char *dst, unsigned char *src, const int width, const int height, const int channels) {
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

__global__ void differenceOnDevice(unsigned char *dst, unsigned char *src, const int width, const int height,
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

void histogramOnHost(unsigned char *dst, unsigned char *src, const int width, const int height, const int channels) {
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

__global__ void histogramOnDevice(unsigned char *dst, unsigned char *src, const int width, const int height,
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

void rotateOnHost(unsigned char *dst, unsigned char *src, const double radian, const int width, const int height,
                  const int channels) {
    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
            // Evaluate the source pixels.
            int x_center = x - round(width / 2.0);
            int y_center = y - round(height / 2.0);
            double xa = x_center * cos(-radian) - y_center * sin(-radian) + round(width / 2.0);
            double ya = x_center * sin(-radian) + y_center * cos(-radian) + round(height / 2.0);

            // Check for out-of-bound coordinates.
            if (xa < 0 or xa > width or ya < 0 or ya > height) {
                // Set pixels to black and exit
                for (int c = 0; c < channels; c++) {
                    int ib = channels * (y * width + x) + c;
                    dst[ib] = 0;
                }

                continue;
            }

            for (int c = 0; c < channels; c++) {
                int ib = channels * (y * width + x) + c;

                // Evaluate the four pixels given xs and ys roundings.
                int ia[4] = {
                        channels * (int(floor(ya)) * width + int(floor(xa))) + c,
                        channels * (int(floor(ya)) * width + int(ceil(xa))) + c,
                        channels * (int(ceil(ya)) * width + int(floor(xa))) + c,
                        channels * (int(ceil(ya)) * width + int(ceil(xa))) + c
                };

                // Evaluate the average value of the destination pixel.
                float sum = 0.0;
                int count = 0;
                for (int k = 0; k < 4; k++) {
                    if (0 <= ia[k] and ia[k] <= width * height * channels) {
                        sum += src[ia[k]];
                        count++;
                    }
                }

                dst[ib] = int(sum / count);
            }
        }
    }
}

__global__ void rotateOnDevice(unsigned char *dst, unsigned char *src, const double radian, const int width,
                               const int height, const int channels) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    // Check for overflow.
    if (i >= width * height) {
        return;
    }

    int x = (int) i / width;
    int y = (i % width);

    // Evaluate the source pixels.
    int x_center = x - round(width / 2.0);
    int y_center = y - round(height / 2.0);
    double xa = x_center * cos(-radian) - y_center * sin(-radian) + round(width / 2.0);
    double ya = x_center * sin(-radian) + y_center * cos(-radian) + round(height / 2.0);

    // Check for out-of-bound coordinates.
    if (xa < 0 or xa > width or ya < 0 or ya > height) {
        // Set pixels to black and exit
        for (int c = 0; c < channels; c++) {
            int ib = channels * (y * width + x) + c;
            dst[ib] = 0;
        }

        return;
    }

    for (int c = 0; c < channels; c++) {
        int ib = channels * (y * width + x) + c;

        // Evaluate the four pixels given xs and ys roundings.
        int ia[4] = {
                channels * (int(floor(ya)) * width + int(floor(xa))) + c,
                channels * (int(floor(ya)) * width + int(ceil(xa))) + c,
                channels * (int(ceil(ya)) * width + int(floor(xa))) + c,
                channels * (int(ceil(ya)) * width + int(ceil(xa))) + c
        };

        // Evaluate the average value of the destination pixel.
        float sum = 0.0;
        int count = 0;
        for (int k = 0; k < 4; k++) {
            if (0 <= ia[k] and ia[k] <= width * height * channels) {
                sum += src[ia[k]];
                count++;
            }
        }

        dst[ib] = int(sum / count);
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