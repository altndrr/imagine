#include "../include/common.h"

#include <stdio.h>
#include <sys/time.h>

#include <cstring>

bool arrayContains(char **list, std::string word) {
    for (int i = 0; i < len(list); i++) {
        if (strcmp(list[i], word.c_str()) == 0) {
            return 1;
        }
    }

    return 0;
}

bool arrayContains(const char **list, std::string word) {
    for (int i = 0; i < len(list); i++) {
        if (strcmp(list[i], word.c_str()) == 0) {
            return 1;
        }
    }

    return 0;
}

double cpuSecond() {
    struct timeval tp;
    gettimeofday(&tp, NULL);
    return ((double)tp.tv_sec + (double)tp.tv_usec * 1.e-6);
}

size_t len(char **array) {
    size_t ret = 0;
    while (*(array++))
        ret += 1;

    return ret;
}

size_t len(const char **array) {
    size_t ret = 0;
    while (*(array++))
        ret += 1;

    return ret;
}

std::string zfill(int num, int size) {
    int nDigits = 0, no = num;
    do {
        no = (int)no / 10;
        nDigits++;
    } while (no != 0);

    int lenPadding = size - nDigits;
    std::string padding;
    for (int k = 0; k < lenPadding; k++)
        padding += '0';

    return padding + std::to_string(num).c_str();
}

void cumsum(float *results, float *values, int dim) {
    results[0] = values[0];

    for (int i = 1; i < dim; i++) {
        results[i] = results[i - 1] + values[i];
    }
}

void movingAverage(float *results, float *values, int dim, int windowSide) {
    int windowMargin = int((windowSide - 1) / 2);

    for (int i = 0; i < dim; i++) {
        float sum = 0.0;
        int rounds = 0;
        int minIndex = std::max(0, i - windowMargin);
        int maxIndex = std::min(dim, i + windowMargin) + 1;

        for (int d = minIndex; d < maxIndex; d++) {
            sum += values[d];
            rounds++;
        }

        results[i] = sum / rounds;
    }
}