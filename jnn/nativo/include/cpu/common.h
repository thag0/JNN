#pragma once

#include "arena.h"

// Arena global jni
extern arena_t mem_arena;

#define MAX_ENTRE(a, b) ((a) > (b) ? (a) : (b))
#define MIN_ENTRE(a, b) ((a) < (b) ? (a) : (b))
