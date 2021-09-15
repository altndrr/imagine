#include "include/main.h"
#include "include/common.h"
#include "include/functions.cuh"
#include "include/image.cuh"
#include <cmath>
#include <stdio.h>

void commandHelp() {
    printf("%s\n", USAGE);
    exit(0);
}

void commandVersion() {
    printf("Version: %s\n", VERSION);
    exit(0);
}

int main(int argc, char *argv[]) {
    cudaFree(0);

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

    // Set options.
    bool grayscale = (arrayContains(argv, "--gray") == 1);
    bool time = (arrayContains(argv, "--time") == 1);
    double startTime = cpuSecond();
    const char *device = (arrayContains(argv, "--cuda") == 1) ? "cuda" : "cpu";
    printf("Device: %s\n", device);

    // Call transpose command.
    if (arrayContains(argv, "transpose") == 1) {
        Image image("data/lena.png", grayscale);
        image.setDevice(device);

        image.transpose();
        if (time)
            printf("Total execution time: %f seconds\n",
                   cpuSecond() - startTime);

        image.save("data/output.png");
        exit(0);
    }

    // Call convolution command.
    if (arrayContains(argv, "convolution") == 1) {
        Image image("data/lena.png", grayscale);
        image.setDevice(device);

        float kernel[] = {0, -1, 0, -1, 4, -1, 0, -1, 0};
        image.convolution(kernel, 3);
        if (time)
            printf("Total execution time: %f seconds\n",
                   cpuSecond() - startTime);

        image.save("data/output.png");
        exit(0);
    }

    // Call difference command.
    if (arrayContains(argv, "difference") == 1) {
        Image image1("data/lena.png", grayscale);
        Image image2("data/output.png", grayscale);
        image1.setDevice(device);

        Image res = image1 - image2;
        if (time)
            printf("Total execution time: %f seconds\n",
                   cpuSecond() - startTime);

        res.save("data/output.png");
        exit(0);
    }

    // Call rotate command.
    if (arrayContains(argv, "rotate") == 1) {
        Image image("data/lena.png", grayscale);
        image.setDevice(device);

        image.rotate(90.0f);
        if (time)
            printf("Total execution time: %f seconds\n",
                   cpuSecond() - startTime);

        image.save("data/output.png");
        exit(0);
    }

    // Call scale command.
    if (arrayContains(argv, "scale") == 1) {
        Image image("data/lena.png", grayscale);
        image.setDevice(device);

        image.scale(0.5);
        if (time)
            printf("Total execution time: %f seconds\n",
                   cpuSecond() - startTime);

        image.save("data/output.png");
        exit(0);
    }

    // Call translate command.
    if (arrayContains(argv, "translate") == 1) {
        Image image("data/lena.png", grayscale);
        image.setDevice(device);

        image.translate(5, 5);
        if (time)
            printf("Total execution time: %f seconds\n",
                   cpuSecond() - startTime);

        image.save("data/output.png");
        exit(0);
    }

    // Call point command.
    if (arrayContains(argv, "point") == 1) {
        Image image("data/lena.png", grayscale);
        image.setDevice(device);

        int color[] = {255, 0, 0};
        int colorSize = sizeof(color) / sizeof(color[0]);

        image.drawPoint(100, 100, 25, color, colorSize);
        if (time)
            printf("Total execution time: %f seconds\n",
                   cpuSecond() - startTime);

        image.save("data/output.png");
        exit(0);
    }

    // Call line command.
    if (arrayContains(argv, "line") == 1) {
        Image image("data/lena.png", grayscale);
        image.setDevice(device);

        int color[] = {255, 0, 0};
        int colorSize = sizeof(color) / sizeof(color[0]);

        image.drawLine(10, 10, 70, 70, 5, color, colorSize);
        if (time)
            printf("Total execution time: %f seconds\n",
                   cpuSecond() - startTime);

        image.save("data/output.png");
        exit(0);
    }

    // Call shi-tomasi "good features to track" command.
    if (arrayContains(argv, "shi") == 1) {
        Image image("data/lena.png");
        image.setDevice(device);

        int maxCorners = 100;
        int *corners = new int[maxCorners];
        image.goodFeaturesToTrack(corners, maxCorners, 0.05, 5.0);
        if (time)
            printf("Total execution time: %f seconds\n",
                   cpuSecond() - startTime);

        int color[] = {0, 255, 0};
        for (int i = 0; i < maxCorners; i++) {
            image.drawPoint(corners[i], 2, color, 3);
        }

        image.save("data/output.png");
        exit(0);
    }

    // Call lucas-kanade "calc optical flow" command.
    if (arrayContains(argv, "lucas") == 1) {
        Image image1("data/lena.png");
        Image image2("data/output.png");

        image1.setDevice(device);
        image2.setDevice(device);

        int maxCorners = 500;
        int *corners = new int[maxCorners];
        image1.goodFeaturesToTrack(corners, maxCorners, 0.05, 5.0);

        int *currCorners = new int[maxCorners];
        image2.calcOpticalFlow(currCorners, &image1, corners, maxCorners, 2);
        if (time)
            printf("Total execution time: %f seconds\n",
                   cpuSecond() - startTime);

        int red[] = {255, 0, 0};
        int green[] = {0, 255, 0};
        int colorSize = sizeof(green) / sizeof(green[0]);
        for (int i = 0; i < maxCorners; i++) {
            image2.drawLine(corners[i], currCorners[i], 1, green, colorSize);
            if (corners[i] != currCorners[i])
                image2.drawPoint(currCorners[i], 1, red, colorSize);
        }

        image2.save("data/output.png");
        exit(0);
    }

    // Call homography command.
    if (arrayContains(argv, "homography") == 1) {
        Image image1("data/lena.png");
        Image image2("data/output.png");

        image1.setDevice(device);
        image2.setDevice(device);

        int maxCorners = 100;
        int *corners = new int[maxCorners];
        image1.goodFeaturesToTrack(corners, maxCorners, 0.05, 5.0);

        int *currCorners = new int[maxCorners];
        image2.calcOpticalFlow(currCorners, &image1, corners, maxCorners, 2);

        float *A = new float[9];
        image2.findHomography(A, currCorners, corners, maxCorners);
        float dx = A[2];
        float dy = A[5];
        float da = atan2(A[3], A[0]);
        image2.rotate(da);
        image2.translate(dx, dy);
        printf("Stabilising image... Parameters: %d, %d, %d\n", int(dx),
               int(dy), int(da));

        if (time)
            printf("Total execution time: %f seconds\n",
                   cpuSecond() - startTime);

        image2.save("data/output.png");
        exit(0);
    }

    // Call stabilise command.
    if (arrayContains(argv, "stabilise") == 1) {
        std::string inputFolder = "data/inputs/";
        std::string outputFolder = "data/outputs/";
        const int n = 75;

        float *translationsX = new float[n];
        float *translationsY = new float[n];
        float *rotations = new float[n];

        for (int i = 0; i < n; i++) {
            std::string filename1 = inputFolder + zfill(i, 6) + ".jpg";
            std::string filename2 = inputFolder + zfill(i + 1, 6) + ".jpg";

            printf("Elaborating %s...\n", filename1.c_str());

            Image *image1 = new Image(filename1.c_str());
            Image *image2 = new Image(filename2.c_str());

            image1->setDevice(device);
            image2->setDevice(device);

            int maxCorners = 200;
            int *corners = new int[maxCorners];
            int *currCorners = new int[maxCorners];
            image1->goodFeaturesToTrack(corners, maxCorners, 0.05, 5.0);
            image2->calcOpticalFlow(currCorners, image1, corners, maxCorners,
                                    2);

            float *A = new float[9];
            image2->findHomography(A, currCorners, corners, maxCorners);

            translationsX[i] = A[2];
            translationsY[i] = A[5];
            rotations[i] = atan2(A[3], A[0]);

            delete[] corners;
            delete[] currCorners;
            delete[] A;
            delete image1;
            delete image2;
        }

        // Evaluate the cumulative sum of the movements.
        float *cumsumX = new float[n];
        float *cumsumY = new float[n];
        float *cumsumTheta = new float[n];
        cumsum(cumsumX, translationsX, n);
        cumsum(cumsumY, translationsY, n);
        cumsum(cumsumTheta, rotations, n);

        float *movAvgX = new float[n];
        float *movAvgY = new float[n];
        float *movAvgTheta = new float[n];

        movingAverage(movAvgX, cumsumX, n);
        movingAverage(movAvgY, cumsumY, n);
        movingAverage(movAvgTheta, cumsumTheta, n);

        for (int i = 0; i < n + 1; i++) {
            std::string filename = inputFolder + zfill(i, 6) + ".jpg";
            std::string output = outputFolder + zfill(i, 6) + ".jpg";
            Image *image = new Image(filename.c_str());

            if (i == 0 or i == n) {
                image->save(output.c_str());
                delete image;
                continue;
            }

            // Evaluate the smooth movements.
            float dx = translationsX[i] + (movAvgX[i] - cumsumX[i]);
            float dy = translationsY[i] + (movAvgY[i] - cumsumY[i]);
            float da = rotations[i] + (movAvgTheta[i] - cumsumTheta[i]);
            dx = round(dx);
            dy = round(dy);
            da = round(da);

            printf("Stabilising %s... Parameters: %d, %d, %d\n",
                   filename.c_str(), int(dx), int(dy), int(da));

            image->rotate(da);
            image->translate(dx, dy);

            image->save(output.c_str());

            delete image;
        }

        if (time)
            printf("Total execution time: %f seconds\n",
                   cpuSecond() - startTime);

        exit(0);
    }

    // If nothing was executed, call help command.
    commandHelp();
}
