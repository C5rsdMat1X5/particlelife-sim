#!/usr/bin/env bash
set -euo pipefail

GLFW_CFLAGS=$(pkg-config --cflags glfw3)
GLFW_LIBS=$(pkg-config --libs glfw3)
CXXFLAGS="-std=c++17 -O3 -march=native -ffast-math -fomit-frame-pointer -flto"
FRAMEWORKS="-framework Metal -framework QuartzCore -framework Foundation -framework AppKit -framework IOKit"

# Compilar TUs en paralelo
c++ $CXXFLAGS $GLFW_CFLAGS -c src/main.cpp     -o build/main.o     &
c++ $CXXFLAGS $GLFW_CFLAGS -c src/metal_sim.mm -o build/metal_sim.o &
wait

# Linkear
c++ $CXXFLAGS build/main.o build/metal_sim.o \
    $FRAMEWORKS $GLFW_LIBS \
    -o build/circles

echo "Build OK → ./build/circles"
