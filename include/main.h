#ifndef IMAGINE_MAIN_H
#define IMAGINE_MAIN_H

const char *VERSION = "0.1.0";
const char *USAGE = "Usage:\n"
                    "  imagine convolution [--cuda | --gray | --time]\n"
                    "  imagine difference [--cuda | --gray | --time]\n"
                    "  imagine rotate [--cuda | --gray | --time]\n"
                    "  imagine scale [--cuda | --gray | --time]\n"
                    "  imagine translate [--cuda | --gray | --time]\n"
                    "  imagine transpose [--cuda | --gray | --time]\n"
                    "  imagine point [--cuda | --gray | --time]\n"
                    "  imagine line [--cuda | --gray | --time]\n"
                    "  imagine shi [--cuda | --gray | --time]\n"
                    "  imagine lucas [--cuda | --gray | --time]\n"
                    "  imagine homography [--cuda | --gray | --time]\n"
                    "  imagine stabilise [--cuda | --gray | --time]\n"
                    "  imagine (-h | --help)\n"
                    "  imagine --version\n"
                    "\n"
                    "Options:\n"
                    "  --cuda        Execute code on GPU.\n"
                    "  --gray        Convert input to grayscale.\n"
                    "  --time        Print the execution time.\n"
                    "  -h --help     Show this screen.\n"
                    "  --version     Show version.\n"
                    "";

#endif // IMAGINE_MAIN_H
