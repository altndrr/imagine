#ifndef IMAGINE_IMAGE_H
#define IMAGINE_IMAGE_H

#define STB_IMAGE_IMPLEMENTATION
#define STB_IMAGE_WRITE_IMPLEMENTATION

class Image {
public:
    Image(const char *filename);

    void dispose();

    int getChannels();

    unsigned int *getElement(int index);

    unsigned int *getElement(int row, int col);

    const char *getFilename();

    int getHeight();

    int getSize();

    int getWidth();

    void save(const char *filename);


private:
    const char *_filename;
    int _width, _height, _channels;
    unsigned char *_data;
};

#endif //IMAGINE_IMAGE_H
