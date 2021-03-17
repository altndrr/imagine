#include "../include/functions.cuh"

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