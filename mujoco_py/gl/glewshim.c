#include <GL/glew.h>
#include "glshim.h"


int usingEGL() {
    return 0;
}

int initOpenGL(int device_id) {
    return 1;
}

int makeOpenGLContextCurrent(int device_id) {
    // Don't need to make context current here, causes issues with large tests
    return 1;
}

int setOpenGLBufferSize(int device_id, int width, int height) {
    if (width > BUFFER_WIDTH || height > BUFFER_HEIGHT) {
        printf("Buffer size too big\n");
        return -1;
    }
    // Noop since we don't support changing the actual buffer
    return 1;
}

void closeOpenGL() {
    if (is_initialized) {
        OSMesaDestroyContext(ctx);
        is_initialized = 0;
    }
}

unsigned int createPBO(int width, int height, int batchSize, int use_short) {
    return 0;
}

void freePBO(unsigned int pixelBuffer) {
}

void copyFBOToPBO(mjrContext* con,
                  unsigned int pbo_rgb, unsigned int pbo_depth,
                  mjrRect viewport, int bufferOffset) {
}

void readPBO(unsigned char *buffer_rgb, unsigned short *buffer_depth,
             unsigned int pbo_rgb, unsigned int pbo_depth,
             int width, int height, int batchSize) {
}
