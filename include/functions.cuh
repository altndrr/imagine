#ifndef IMAGINE_FUNCTIONS_Hconv
#define IMAGINE_FUNCTIONS_H

#include <cuda_runtime.h>

void transposeOnHost(unsigned char *data, const int width, const int height, const int channels);

__global__ void transposeOnDevice(unsigned char *data, const int width, const int height, const int channels);


#endif //IMAGINE_FUNCTIONS_H
