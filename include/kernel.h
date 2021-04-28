#ifndef IMAGINE_KERNEL_H
#define IMAGINE_KERNEL_H


class Kernel {
public:
    static void Gaussian(float **kernel, int *kernelSide = nullptr);

    static void SobelX(float **kernel, int *kernelSide = nullptr);

    static void SobelY(float **kernel, int *kernelSide = nullptr);

private:
    Kernel() { ; }
};


#endif //IMAGINE_KERNEL_H
