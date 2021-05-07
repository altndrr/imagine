#include "include/main.h"

#include <stdio.h>

#include <string>

#include "include/common.h"
#include "include/image.cuh"
#include "include/kernel.h"

void commandHelp() {
    printf("%s\n", USAGE);
    exit(0);
}

void commandVersion() {
    printf("Version: %s\n", VERSION);
    exit(0);
}

int main(int argc, char *argv[]) {
    // If no argument is passed, call help command.
    if (argc == 1) {
        commandHelp();
    }

    // Call help command.
    if (arrayContains(argv, "--help") == 1 or arrayContains(argv, "-h") == 1) {
        commandHelp();
    }

    // Call version command.
    if (arrayContains(argv, "--version") == 1) {
        commandVersion();
    }

    // If nothing was executed, call help command.
    commandHelp();
}
