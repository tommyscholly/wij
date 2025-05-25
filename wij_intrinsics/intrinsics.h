#ifndef INTRINSICS_H
#define INTRINSICS_H

#include <stdint.h>
#include <stdio.h>

void write_int(int32_t x);

// str functions
typedef struct {
    const char *data;
    size_t len;
} StringData;

StringData *str_alloc(size_t len);
void str_free(StringData *str);

// gc functions
typedef struct {
    size_t gcbits;
    void *data;
} GcData;

#endif
