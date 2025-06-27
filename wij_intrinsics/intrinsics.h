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
StringData *int_to_string(int32_t x);
void str_concat(StringData *a, StringData *b);
void str_free(StringData *str);
void print_str(StringData *str);

#endif
