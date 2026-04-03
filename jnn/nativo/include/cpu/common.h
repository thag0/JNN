#pragma once

#include "arena.h"
#define ARENA_INITIAL_CAP_MB 256

// Arena global jni
extern arena_t arena;

#define MAX_ENTRE(a, b) ((a) > (b) ? (a) : (b))
#define MIN_ENTRE(a, b) ((a) < (b) ? (a) : (b))
