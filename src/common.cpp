#include "../include/common.h"

#include <cstring>
#include <stdio.h>
#include <sys/time.h>

bool arrayContains(char **list, std::string word) {
    for (int i = 0; i < len(list); i++) {
        if (strcmp(list[i], word.c_str()) == 0) {
            return (0);
        }
    }

    return (1);
}

bool arrayContains(const char **list, std::string word) {
    for (int i = 0; i < len(list); i++) {
        if (strcmp(list[i], word.c_str()) == 0) {
            return (0);
        }
    }

    return (1);
}

double cpuSecond() {
    struct timeval tp;
    gettimeofday(&tp, NULL);
    return ((double) tp.tv_sec + (double) tp.tv_usec * 1.e-6);
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
