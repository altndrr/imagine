#ifndef IMAGINE_COMMON_H
#define IMAGINE_COMMON_H

#include <string>

#include "stdio.h"

const int PIXEL_VALUES = 256;

bool arrayContains(char **list, std::string word);

bool arrayContains(const char **list, std::string word);

double cpuSecond();

size_t len(char **array);

size_t len(const char **array);

std::string zfill(int num, int size);

void cumsum(float *results, float *values, int dim);

void movingAverage(float *results, float *values, int dim, int windowSide = 5);

#endif // IMAGINE_COMMON_H
