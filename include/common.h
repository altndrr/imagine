#ifndef IMAGINE_COMMON_H
#define IMAGINE_COMMON_H

#include "stdio.h"
#include <string>

bool arrayContains(char **list, std::string word);

bool arrayContains(const char **list, std::string word);

double cpuSecond();

size_t len(char **array);

size_t len(const char **array);

#endif //IMAGINE_COMMON_H
