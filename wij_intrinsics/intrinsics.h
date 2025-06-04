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

StringData *make_string(const char *data, size_t len);
void str_print(StringData *str);
void str_free(StringData *str);

// gc functions
typedef struct {
    size_t gcbits;
    void *data;
} GcData;

#endif
