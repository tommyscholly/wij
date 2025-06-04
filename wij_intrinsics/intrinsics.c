#include "intrinsics.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

void write_int(int32_t x) { printf("%d\n", x); }

StringData *make_string(const char *data, size_t len) {
    StringData *str = malloc(sizeof(StringData));
    str->data = data;
    str->len = strlen(data);
    return str;
}

void print_string(StringData *str) {
    write(1, str->data, str->len);
    write(1, "\n", 1);
}
