#include "../include/kernel.h"

void Kernel::Gaussian(float **kernel, int *kernelSide) {
    const int side = 3;

    if (kernelSide != nullptr) {
        *kernelSide = side;
    }

    *kernel = new float[side * side]{0.0625, 0.125, 0.0625, 0.125, 0.250, 0.125, 0.0625, 0.125, 0.0625};
}

void Kernel::SobelX(float **kernel, int *kernelSide) {
    const int side = 3;

    if (kernelSide != nullptr) {
        *kernelSide = side;
    }

    *kernel = new float[side * side]{0.111, 0, -0.111, 0.222, 0, -0.222, 0.111, 0, -0.111};
}

void Kernel::SobelY(float **kernel, int *kernelSide) {
    const int side = 3;

    if (kernelSide != nullptr) {
        *kernelSide = side;
    }
    *kernel = new float[side * side]{0.111, 0.222, 0.111, 0, 0, 0, -0.111, -0.222, -0.111};
}