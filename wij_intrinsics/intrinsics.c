#include "intrinsics.h"
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

StringData *int_to_string(int32_t x) {
    char buf[32];
    sprintf(buf, "%d", x);
    return make_string(buf, strlen(buf));
}

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

void str_concat(StringData *a, StringData *b) {
    a->data = realloc((void *)a->data, a->len + b->len);
    memcpy((void *)a->data + a->len, b->data, b->len);
    a->len += b->len;
}

void str_free(StringData *str) {
    free((void *)str->data);
    free(str);
}
