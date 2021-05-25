#include <bits/stdc++.h>
#include <cmath>

#include "../include/common.h"
#include "../include/functions.cuh"

void convolutionOnHost(unsigned char *dst, unsigned char *src, float *kernel,
                       int kernelSide, const int width, const int height,
                       const int channels) {
    unsigned int margin = int((kernelSide - 1) / 2);

    // Loop through each pixel.
    for (int y = margin; y < width - margin; y++) {
        for (int x = margin; x < height - margin; x++) {
            // Loop through each element of the kernel.
            for (int dy = 0; dy < kernelSide; dy++) {
                for (int dx = 0; dx < kernelSide; dx++) {
                    // Loop through the channels of the image.
                    for (int c = 0; c < channels; c++) {
                        int src_i = channels * ((x + (dx - margin)) * width +
                                                (y + (dy - margin))) +
                                    c;
                        int ker_i = dx * kernelSide + dy;
                        int dst_i = channels * (x * width + y) + c;

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

__global__ void convolutionOnDevice(unsigned char *dst, unsigned char *src,
                                    float *kernel, int kernelSide,
                                    const int width, const int height,
                                    const int channels) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    // Check for overflow.
    if (i >= width * height) {
        return;
    }

    unsigned int margin = int((kernelSide - 1) / 2);

    int x = (int)i / width;
    int y = (i % width);

    // Check for minimum padding.
    if (y < margin or y > width - margin - 1 or x < margin or
        x > height - margin - 1) {
        return;
    }

    // Loop through each element of the kernel.
    for (int dy = 0; dy < kernelSide; dy++) {
        for (int dx = 0; dx < kernelSide; dx++) {
            // Loop through the channels of the image.
            for (int c = 0; c < channels; c++) {
                int src_i = channels * ((x + (dx - margin)) * width +
                                        (y + (dy - margin))) +
                            c;
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

void drawLineOnHost(unsigned char *data, int x1, int y1, int x2, int y2,
                    int radius, int *color, int colorSize, int width,
                    int height, int channels) {
    for (int dy = min(y1, y2); dy < max(y1, y2); dy++) {
        for (int dx = min(x1, x2); dx < max(x1, x2); dx++) {
            int interpolatedY = (y1 * (x2 - dx) + y2 * (dx - x1)) / (x2 - x1);

            if (interpolatedY - radius > dy or interpolatedY + radius < dy) {
                continue;
            }

            int index = (dx * width + dy) * channels;

            for (int c = 0; c < min(channels, colorSize); c++) {
                if (index + c < width * height * channels) {
                    data[index + c] = color[c];
                }
            }
        }
    }
}

__global__ void drawLineOnDevice(unsigned char *data, int x1, int y1, int x2,
                                 int y2, int radius, int *color, int colorSize,
                                 int width, int height, int channels) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    // Check for overflow.
    if (i >= width * height) {
        return;
    }

    int dx = (int)i / width;
    int dy = (i % width);

    // Check for boundaries.
    int interpolatedY = (y1 * (x2 - dx) + y2 * (dx - x1)) / (x2 - x1);
    if (dx < min(x1, x2) or dx >= max(x1, x2) or dy < min(y1, y2) or
        dy >= max(y1, y2) or interpolatedY - radius > dy or
        interpolatedY + radius < dy) {
        return;
    }

    for (int c = 0; c < min(channels, colorSize); c++) {
        int index = channels * i;
        if (index + c < width * height * channels) {
            data[index + c] = color[c];
        }
    }
}

void drawPointOnHost(unsigned char *data, int x, int y, int radius, int *color,
                     int colorSize, int width, int height, int channels) {
    for (int dy = max(0, y - radius); dy < y + radius; dy++) {
        for (int dx = max(0, x - radius); dx < x + radius; dx++) {
            int index = (dx * width + dy) * channels;

            for (int c = 0; c < min(channels, colorSize); c++) {
                if (index + c < width * height * channels) {
                    data[index + c] = color[c];
                }
            }
        }
    }
}

__global__ void drawPointOnDevice(unsigned char *data, int x, int y, int radius,
                                  int *color, int colorSize, int width,
                                  int height, int channels) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    // Check for overflow.
    if (i >= width * height) {
        return;
    }

    int dx = (int)i / width;
    int dy = (i % width);

    // Check for point boundaries.
    if (dy < y - radius or dy >= y + radius or dx < x - radius or
        dx >= x + radius) {
        return;
    }

    for (int c = 0; c < min(channels, colorSize); c++) {
        int index = channels * i;
        if (index + c < width * height * channels) {
            data[index + c] = color[c];
        }
    }
}

void differenceOnHost(unsigned char *dst, unsigned char *src, const int width,
                      const int height, const int channels) {
    for (int y = 0; y < width; y++) {
        for (int x = 0; x < height; x++) {
            for (int c = 0; c < channels; c++) {
                int i = channels * (x * width + y) + c;
                if (dst[i] > src[i]) {
                    dst[i] = dst[i] - src[i];
                } else {
                    dst[i] = src[i] - dst[i];
                }
            }
        }
    }
}

__global__ void differenceOnDevice(unsigned char *dst, unsigned char *src,
                                   const int width, const int height,
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

void cornerScoreOnHost(unsigned char *gradX, unsigned char *gradY, float *R,
                       int width, int height) {
    const int windowSide = 3;
    const int windowMargin = int((windowSide - 1) / 2);

    for (int i = 0; i < width * height; i++) {
        int x = (int)i / width;
        int y = (i % width);

        // Check for out-of-bound coordinates.
        R[i] = 0;
        if (x < windowMargin or y < windowMargin or
            x > height - windowMargin - 1 or y > width - windowMargin - 1) {
            continue;
        }

        // Create the windows Ix and Iy.
        float *Ix = new float[windowSide * windowSide];
        float *Iy = new float[windowSide * windowSide];
        for (int wi = 0; wi < windowSide * windowSide; wi++) {
            int dx = ((int)wi / windowSide) - windowMargin;
            int dy = (wi % windowSide) - windowMargin;
            int di = (x + dx) * width + (y + dy);

            Ix[wi] = (float)gradX[di] / PIXEL_VALUES;
            Iy[wi] = (float)gradY[di] / PIXEL_VALUES;
        }

        // Construct the structural matrix.
        float *M = new float[4];
        sumOfMatmulOnHost(&M[0], Ix, Ix, windowSide);
        sumOfMatmulOnHost(&M[1], Ix, Iy, windowSide);
        sumOfMatmulOnHost(&M[2], Iy, Ix, windowSide);
        sumOfMatmulOnHost(&M[3], Iy, Iy, windowSide);

        // Evaluate the pixel score.
        float m = (M[0] + M[3]) / 2;
        float p = (M[0] * M[3]) - (M[1] * M[2]);
        float lambda1 = m + sqrt(m * m - p);
        float lambda2 = m - sqrt(m * m - p);
        R[i] = min(lambda1, lambda2);
    }
}

__global__ void cornerScoreOnDevice(unsigned char *gradX, unsigned char *gradY,
                                    float *R, int width, int height) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    const int windowSide = 3;
    const int windowMargin = int((windowSide - 1) / 2);

    // Check for overflow.
    if (i >= width * height) {
        return;
    }

    int x = (int)i / width;
    int y = (i % width);

    // Check for out-of-bound coordinates.
    R[i] = 0;
    if (x < windowMargin or y < windowMargin or x > height - windowMargin - 1 or
        y > width - windowMargin - 1) {
        return;
    }

    // Create the windows Ix and Iy.
    float *Ix = new float[windowSide * windowSide];
    float *Iy = new float[windowSide * windowSide];
    for (int wi = 0; wi < windowSide * windowSide; wi++) {
        int dx = ((int)wi / windowSide) - windowMargin;
        int dy = (wi % windowSide) - windowMargin;
        int di = (x + dx) * width + (y + dy);

        Ix[wi] = (float)gradX[di] / PIXEL_VALUES;
        Iy[wi] = (float)gradY[di] / PIXEL_VALUES;
    }

    // Construct the structural matrix.
    float *M = new float[4]{0, 0, 0, 0};
    sumOfMatmulOnDevice(&M[0], Ix, Ix, windowSide);
    sumOfMatmulOnDevice(&M[1], Ix, Iy, windowSide);
    sumOfMatmulOnDevice(&M[2], Iy, Ix, windowSide);
    sumOfMatmulOnDevice(&M[3], Iy, Iy, windowSide);

    delete[] Ix;
    delete[] Iy;

    // Evaluate the pixel score.
    float m = (M[0] + M[3]) / 2;
    float p = (M[0] * M[3]) - (M[1] * M[2]);
    float lambda1 = m + sqrt(m * m - p);
    float lambda2 = m - sqrt(m * m - p);
    R[i] = min(lambda1, lambda2);

    delete[] M;
}

void histogramOnHost(unsigned char *dst, unsigned char *src, const int width,
                     const int height, const int channels) {
    for (int y = 0; y < width; y++) {
        for (int x = 0; x < height; x++) {
            for (int c = 0; c < channels; c++) {
                int ia = channels * (y * width + x) + c;
                int ib = PIXEL_VALUES * c + int(src[ia]);

                dst[ib] += 1;
            }
        }
    }
}

__global__ void histogramOnDevice(unsigned char *dst, unsigned char *src,
                                  const int width, const int height,
                                  const int channels) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    // Check for overflow.
    if (i > PIXEL_VALUES * channels) {
        return;
    }

    for (int y = 0; y < width; y++) {
        for (int x = 0; x < height; x++) {
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

void opticalFLowOnHost(int *currentCorners, int *corners, int maxCorners,
                       unsigned char **currPyramidalScales,
                       unsigned char **prevPyramidalScales, int levels,
                       int width0, int height0) {
    const int patchSide = 5;
    const int windowSide = 9;
    const int windowMargin = int((windowSide - 1) / 2);
    unsigned char *prevPatch = new unsigned char[patchSide * patchSide];
    unsigned char *currPatch = new unsigned char[patchSide * patchSide];

    for (int l = levels - 1; l >= 0; l--) {
        int width = width0 / pow(2, l);
        int height = height0 / pow(2, l);
        float minSse;

        for (int i = 0; i < maxCorners; i++) {
            // Downscale corner from the previous frame.
            int lx = (corners[i] / width0) * pow(2, -l);
            int ly = (corners[i] % width0) * pow(2, -l);

            int prevCorner = int(lx * width + ly);
            minSse = 100;

            if (l == levels - 1) {
                currentCorners[i] = prevCorner;
            } else {
                // Upscale corner from the previous layer.
                int ux = int(currentCorners[i] / (width * 0.5)) * 2;
                int uy = (currentCorners[i] % int((width * 0.5))) * 2;
                currentCorners[i] = int(ux * width + uy);
            }

            extractPatchOnHost(prevPatch, prevPyramidalScales[l], prevCorner,
                               patchSide, width, height);

            int x = (int)currentCorners[i] / width;
            int y = currentCorners[i] % width;
            for (int wi = 0; wi < windowSide * windowSide; wi++) {
                int dx = ((int)wi / windowSide) - windowMargin;
                int dy = (wi % windowSide) - windowMargin;
                int di = (x + dx) * width + (y + dy);

                extractPatchOnHost(currPatch, currPyramidalScales[l], di,
                                   patchSide, width, height);

                float sse = sumOfSquareDifferencesOnHost(prevPatch, currPatch,
                                                         patchSide);

                if (sse < minSse) {
                    currentCorners[i] = di;
                    minSse = sse;
                }
            }
        }
    }
}

__global__ void opticalFLowOnDevice(int *currentCorners, int *corners,
                                    int maxCorners,
                                    unsigned char *currPyramidalScales,
                                    unsigned char *prevPyramidalScales,
                                    int levels, int offsetSize, int width0,
                                    int height0) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    const int patchSide = 5;
    const int windowSide = 9;
    const int windowMargin = int((windowSide - 1) / 2);
    unsigned char *prevPatch = new unsigned char[patchSide * patchSide];
    unsigned char *currPatch = new unsigned char[patchSide * patchSide];

    for (int l = levels - 1; l >= 0; l--) {
        int width = width0 / pow(2, l);
        int height = height0 / pow(2, l);
        float minSse = 100;

        // Downscale corner from the previous frame.
        int lx = (corners[i] / width0) * pow(2, -l);
        int ly = (corners[i] % width0) * pow(2, -l);

        int prevCorner = int(lx * width + ly);

        if (l == levels - 1) {
            currentCorners[i] = prevCorner;
        } else {
            // Upscale corner from the previous layer.
            int ux = int(currentCorners[i] / (width * 0.5)) * 2;
            int uy = (currentCorners[i] % int((width * 0.5))) * 2;
            currentCorners[i] = int(ux * width + uy);
        }

        extractPatchOnDevice(prevPatch, prevPyramidalScales + l * offsetSize,
                             prevCorner, patchSide, width, height);

        int x = (int)currentCorners[i] / width;
        int y = currentCorners[i] % width;
        for (int wi = 0; wi < windowSide * windowSide; wi++) {
            int dx = ((int)wi / windowSide) - windowMargin;
            int dy = (wi % windowSide) - windowMargin;
            int di = (x + dx) * width + (y + dy);

            extractPatchOnDevice(currPatch,
                                 currPyramidalScales + l * offsetSize, di,
                                 patchSide, width, height);

            float sse =
                sumOfSquareDifferencesOnDevice(prevPatch, currPatch, patchSide);

            if (sse < minSse) {
                currentCorners[i] = di;
                minSse = sse;
            }
        }
    }

    delete[] prevPatch;
    delete[] currPatch;
}

void rotateOnHost(unsigned char *dst, unsigned char *src, const double radian,
                  const int width, const int height, const int channels) {
    for (int y = 0; y < width; y++) {
        for (int x = 0; x < height; x++) {
            // Evaluate the source pixels.
            int x_center = x - round(height / 2.0);
            int y_center = y - round(width / 2.0);
            double xa = x_center * cos(-radian) - y_center * sin(-radian) +
                        round(height / 2.0);
            double ya = x_center * sin(-radian) + y_center * cos(-radian) +
                        round(width / 2.0);

            // Check for out-of-bound coordinates.
            if (xa < 0 or xa > height or ya < 0 or ya > width) {
                // Set pixels to black and exit
                for (int c = 0; c < channels; c++) {
                    int ib = channels * (x * width + y) + c;
                    dst[ib] = 0;
                }

                continue;
            }

            for (int c = 0; c < channels; c++) {
                int ib = channels * (x * width + y) + c;

                // Evaluate the four pixels given xs and ys roundings.
                int ia[4] = {
                    channels * (int(floor(xa)) * width + int(floor(ya))) + c,
                    channels * (int(floor(xa)) * width + int(ceil(ya))) + c,
                    channels * (int(ceil(xa)) * width + int(floor(ya))) + c,
                    channels * (int(ceil(xa)) * width + int(ceil(ya))) + c};

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

__global__ void rotateOnDevice(unsigned char *dst, unsigned char *src,
                               const double radian, const int width,
                               const int height, const int channels) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    // Check for overflow.
    if (i >= width * height) {
        return;
    }

    int x = (int)i / width;
    int y = (i % width);

    // Evaluate the source pixels.
    int x_center = x - round(height / 2.0);
    int y_center = y - round(width / 2.0);
    double xa =
        x_center * cos(-radian) - y_center * sin(-radian) + round(height / 2.0);
    double ya =
        x_center * sin(-radian) + y_center * cos(-radian) + round(width / 2.0);

    // Check for out-of-bound coordinates.
    if (xa < 0 or xa > height or ya < 0 or ya > width) {
        // Set pixels to black and exit
        for (int c = 0; c < channels; c++) {
            int ib = channels * (x * width + y) + c;
            dst[ib] = 0;
        }

        return;
    }

    for (int c = 0; c < channels; c++) {
        int ib = channels * (x * width + y) + c;

        // Evaluate the four pixels given xs and ys roundings.
        int ia[4] = {channels * (int(floor(xa)) * width + int(floor(ya))) + c,
                     channels * (int(floor(xa)) * width + int(ceil(ya))) + c,
                     channels * (int(ceil(xa)) * width + int(floor(ya))) + c,
                     channels * (int(ceil(xa)) * width + int(ceil(ya))) + c};

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

void scaleOnHost(unsigned char *dst, unsigned char *src, const double ratio,
                 const int width, const int height, const int channels) {
    int newWidth = width * ratio;
    int newHeight = height * ratio;
    float inverseRatio = 1.0 / ratio;

    for (int y = 0; y < newWidth; y++) {
        for (int x = 0; x < newHeight; x++) {
            for (int c = 0; c < channels; c++) {
                int i = (x * newWidth + y) * channels + c;
                float tempValue = 0.0;

                for (int dy = -1; dy < 2; dy++) {
                    for (int dx = -1; dx < 2; dx++) {
                        int oldI = ((int(inverseRatio * x) + dx) * width +
                                    (int(inverseRatio * y) + dy)) *
                                       channels +
                                   c;
                        float weight = 1 / (pow(2, 2 + abs(dx) + abs(dy)));

                        if (oldI < 0 or oldI > width * height * channels) {
                            continue;
                        }
                        tempValue += weight * src[oldI];
                    }
                }
                dst[i] = tempValue;
            }
        }
    }
}

__global__ void scaleOnDevice(unsigned char *dst, unsigned char *src,
                              const double ratio, const int width,
                              const int height, const int channels) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    int newWidth = width * ratio;
    int newHeight = height * ratio;
    float inverseRatio = 1.0 / ratio;

    // Check for overflow.
    if (i > newWidth * newHeight) {
        return;
    }

    int x = (int)i / newWidth;
    int y = (i % newWidth);

    for (int c = 0; c < channels; c++) {
        float tempValue = 0.0;

        for (int dy = -1; dy < 2; dy++) {
            for (int dx = -1; dx < 2; dx++) {
                int src_i = ((int(inverseRatio * x) + dx) * width +
                             (int(inverseRatio * y) + dy)) *
                                channels +
                            c;
                float weight = 1 / (pow(2, 2 + abs(dx) + abs(dy)));

                if (src_i < 0 or src_i > width * height * channels) {
                    continue;
                }
                tempValue += weight * src[src_i];
            }
        }
        dst[i * channels + c] = tempValue;
    }
}

void translateOnHost(unsigned char *dst, unsigned char *src, int px, int py,
                     const int width, const int height, const int channels) {
    for (int y = 0; y < width; y++) {
        for (int x = 0; x < height; x++) {
            // Evaluate the source pixels.
            int xa = x - px;
            int ya = y - py;

            // Check for out-of-bound coordinates.
            if (xa < 0 or xa > height or ya < 0 or ya > width) {
                // Set pixels to black and exit
                for (int c = 0; c < channels; c++) {
                    int ib = channels * (x * width + y) + c;
                    dst[ib] = 0;
                }

                continue;
            }

            for (int c = 0; c < channels; c++) {
                int ia = channels * (xa * width + ya) + c;
                int ib = channels * (x * width + y) + c;
                dst[ib] = src[ia];
            }
        }
    }
}

__global__ void translateOnDevice(unsigned char *dst, unsigned char *src,
                                  int px, int py, const int width,
                                  const int height, const int channels) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    // Check for overflow.
    if (i >= width * height) {
        return;
    }

    int x = (int)i / width;
    int y = (i % width);

    // Evaluate the source pixels.
    int xa = x - px;
    int ya = y - py;

    // Check for out-of-bound coordinates.
    if (xa < 0 or xa > height or ya < 0 or ya > width) {
        // Set pixels to black and exit.
        for (int c = 0; c < channels; c++) {
            int ib = channels * (x * width + y) + c;
            dst[ib] = 0;
        }

        return;
    }

    for (int c = 0; c < channels; c++) {
        int ia = channels * (xa * width + ya) + c;
        int ib = channels * (x * width + y) + c;
        dst[ib] = src[ia];
    }
}

void transposeOnHost(unsigned char *data, const int width, const int height,
                     const int channels) {
    for (int y = 0; y < width; y++) {
        for (int x = 0; x < height; x++) {
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

__global__ void transposeOnDevice(unsigned char *data, const int width,
                                  const int height, const int channels) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    // Check for overflow.
    if (i >= width * height) {
        return;
    }

    for (int c = 0; c < channels; c++) {
        int ia = channels * i + c;
        int ib = channels * ((i % width) * height + ((int)i / width)) + c;

        if (ia > ib) {
            continue;
        }

        unsigned char temp = data[ib];
        data[ib] = data[ia];
        data[ia] = temp;
    }
}

void sumOfMatmulOnHost(float *total, float *A, float *B, int side) {
    float *C = new float[side * side];
    *total = 0;

    for (int i = 0; i < side * side; i++) {
        int x = (int)i / side;
        int y = (i % side);

        C[i] = 0;
        for (int d = 0; d < side; d++) {
            int ia = x * side + d;
            int ib = d * side + y;

            C[i] += A[ia] * B[ib];
        }

        *total += C[i];
    }
}

__device__ void sumOfMatmulOnDevice(float *total, float *A, float *B,
                                    int side) {
    *total = 0;

    for (int i = 0; i < side * side; i++) {
        int x = (int)i / side;
        int y = (i % side);

        for (int d = 0; d < side; d++) {
            int ia = x * side + d;
            int ib = d * side + y;

            *total += A[ia] * B[ib];
        }
    }
}

float sumOfSquareDifferencesOnHost(unsigned char *patch1, unsigned char *patch2,
                                   int patchSide) {
    float sse = 0.0;
    for (int i = 0; i < patchSide * patchSide; i++) {
        sse += pow(float(patch1[i] - patch2[i]), 2);
    }

    return sse;
}

__device__ float sumOfSquareDifferencesOnDevice(unsigned char *patch1,
                                                unsigned char *patch2,
                                                int patchSide) {
    float sse = 0.0;
    for (int i = 0; i < patchSide * patchSide; i++) {
        sse += pow(float(patch1[i] - patch2[i]), 2);
    }

    return sse;
}

void extractPatchOnHost(unsigned char *patch, unsigned char *data,
                        int centerIndex, int patchSide, int width, int height) {
    const int patchMargin = int((patchSide - 1) / 2);

    for (int pi = 0; pi < patchSide * patchSide; pi++) {
        int x = (int)centerIndex / width;
        int y = centerIndex % width;
        int dx = ((int)pi / patchSide) - patchMargin;
        int dy = (pi % patchSide) - patchMargin;
        int di = (x + dx) * width + (y + dy);

        if (di < 0 or di > width * height) {
            patch[pi] = 0;
        } else {
            patch[pi] = data[di];
        }
    }
}

__device__ void extractPatchOnDevice(unsigned char *patch, unsigned char *data,
                                     int centerIndex, int patchSide, int width,
                                     int height) {
    const int patchMargin = int((patchSide - 1) / 2);

    for (int pi = 0; pi < patchSide * patchSide; pi++) {
        int x = (int)centerIndex / width;
        int y = centerIndex % width;
        int dx = ((int)pi / patchSide) - patchMargin;
        int dy = (pi % patchSide) - patchMargin;
        int di = (x + dx) * width + (y + dy);

        if (di < 0 or di > width * height) {
            patch[pi] = 0;
        } else {
            patch[pi] = data[di];
        }
    }
}

void findHomographyRANSACOnHost(float *matrices, float *scores, int maxIter,
                                int *currentCorners, int *previousCorners,
                                int maxCorners, int width, int height,
                                float thresholdError, float minConfidence) {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<> uniform(0, maxCorners);

    const int n_points = 3;
    const int space_dim = 2;

    // Create maxIter models.
    float *srcTriplet = new float[n_points * space_dim];
    float *dstTriplet = new float[n_points * space_dim];
    float *estPoint = new float[space_dim];
    float *srcPoint = new float[space_dim];
    float *dstPoint = new float[space_dim];
    for (int n = 0; n < maxIter; n++) {
        int offset = n * (n_points * (space_dim + 1));
        scores[n] = INFINITY;

        // Select the minimum number of data points to estimate a model.
        for (int k = 0; k < n_points; k++) {
            int i = uniform(gen);
            srcTriplet[k * space_dim] = (int)previousCorners[i] / width;
            srcTriplet[k * space_dim + 1] = previousCorners[i] % width;
            dstTriplet[k * space_dim] = (int)currentCorners[i] / width;
            dstTriplet[k * space_dim + 1] = currentCorners[i] % width;
        }

        // Estimate the model that fit the hypothetical inliers.
        estimateTransformOnHost(matrices + offset, srcTriplet, dstTriplet);

        // Count the points that fit the model and the total error.
        int nInliers = 0;
        float totalError = 0.0;
        for (int i = 0; i < maxCorners; i++) {
            srcPoint[0] = (int)previousCorners[i] / width;
            srcPoint[1] = previousCorners[i] % width;
            dstPoint[0] = (int)currentCorners[i] / width;
            dstPoint[1] = currentCorners[i] % width;

            // Apply the transform and evaluate the error.
            applyTransformOnHost(estPoint, srcPoint, matrices + offset);
            float reprojError = pow(int(estPoint[0] - dstPoint[0]), 2) +
                                pow(int(estPoint[1] - dstPoint[1]), 2);
            nInliers += int(reprojError < thresholdError);
            totalError += reprojError;
        }

        // Set the matrix score to the error if the confidence is high
        // enough.
        float confidence = (float)nInliers / maxCorners;
        if (confidence >= minConfidence) {
            scores[n] = totalError;
        }
    }

    delete[] srcTriplet;
    delete[] dstTriplet;
    delete[] estPoint;
    delete[] srcPoint;
    delete[] dstPoint;
}

void estimateTransformOnHost(float *A, float *Ui, float *vi) {
    const int n_points = 3;
    const int space_dim = 2;

    // Create X and Y matrices.
    float *X = new float[n_points * (space_dim + 1)];
    float *Y = new float[n_points * (space_dim + 1)];
    for (int d = 0; d < space_dim + 1; d++) {
        for (int n = 0; n < n_points; n++) {
            int i = d * (n_points) + n;
            int j = n * (space_dim) + d;

            if (d == space_dim) {
                X[i] = 1;
                Y[i] = int(n >= n_points - 1);
            } else {
                X[i] = Ui[j];
                Y[i] = vi[j];
            }
        }
    }

    float *Xi = new float[n_points * (space_dim + 1)];
    invert3x3MatrixOnHost(Xi, X);

    // Get the affine transformation matrix.
    matmulOnHost(A, Y, Xi, n_points);

    delete[] X;
    delete[] Y;
    delete[] Xi;
}

void invert3x3MatrixOnHost(float *Xi, float *X) {
    float det = X[0] * (X[4] * X[8] - X[5] * X[7]) -
                X[1] * (X[3] * X[8] - X[5] * X[6]) +
                X[2] * (X[3] * X[7] - X[4] * X[6]);

    Xi[0] = +float(X[4] * X[8] - X[5] * X[7]) / det;
    Xi[1] = -float(X[1] * X[8] - X[2] * X[7]) / det;
    Xi[2] = +float(X[1] * X[5] - X[2] * X[4]) / det;
    Xi[3] = -float(X[3] * X[8] - X[5] * X[6]) / det;
    Xi[4] = +float(X[0] * X[8] - X[2] * X[6]) / det;
    Xi[5] = -float(X[0] * X[5] - X[2] * X[3]) / det;
    Xi[6] = +float(X[3] * X[7] - X[4] * X[6]) / det;
    Xi[7] = -float(X[0] * X[7] - X[1] * X[6]) / det;
    Xi[8] = +float(X[0] * X[4] - X[1] * X[3]) / det;
}

void matmulOnHost(float *C, float *A, float *B, int side) {
    for (int i = 0; i < side * side; i++) {
        int x = (int)i / side;
        int y = (i % side);

        C[i] = 0;
        for (int d = 0; d < side; d++) {
            int ia = x * side + d;
            int ib = d * side + y;

            C[i] += A[ia] * B[ib];
        }
    }
}

void applyTransformOnHost(float *dst, float *src, float *A) {
    const int space_dim = 2;

    for (int i = 0; i < space_dim; i++) {
        dst[i] = 0.0;
        dst[i] += src[0] * A[i * 3 + 0];
        dst[i] += src[1] * A[i * 3 + 1];
    }
}