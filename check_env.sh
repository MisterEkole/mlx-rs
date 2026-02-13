#!/bin/bash

# --- Color Definitions ---
RED='\033[0;31m'
GREEN='\033[0;32m'
NC='\033[0m' # No Color

echo "🔍 MLX-RS Pre-flight Environment Check..."

# 1. Check if MLX_C_PATH is set
if [ -z "$MLX_C_PATH" ]; then
    echo -e "${RED}❌ ERROR: MLX_C_PATH is not set.${NC}"
    echo "Please set it to the root of your mlx-c directory."
    echo "Example: export MLX_C_PATH=/Users/$(whoami)/mlx-c"
    exit 1
fi

# 2. Check if the directory exists
if [ ! -d "$MLX_C_PATH" ]; then
    echo -e "${RED}❌ ERROR: MLX_C_PATH directory does not exist: $MLX_C_PATH${NC}"
    exit 1
fi

# 3. Check for static libraries in build folders
# We expect libmlxc.a in /build and libmlx.a in /build/_deps/mlx-build
MLXC_LIB="$MLX_C_PATH/build/libmlxc.a"
MLX_ENGINE_LIB="$MLX_C_PATH/build/_deps/mlx-build/libmlx.a"

MISSING=0

if [ ! -f "$MLXC_LIB" ]; then
    echo -e "${RED}⚠️  Missing: $MLXC_LIB${NC}"
    MISSING=1
fi

if [ ! -f "$MLX_ENGINE_LIB" ]; then
    echo -e "${RED}⚠️  Missing: $MLX_ENGINE_LIB${NC}"
    MISSING=1
fi

if [ $MISSING -eq 1 ]; then
    echo -e "${RED}❌ ERROR: MLX binaries not found. Did you run 'make' inside the mlx-c/build directory?${NC}"
    exit 1
fi

# 4. Architecture Check (Optional but recommended for Apple Silicon)
ARCH=$(uname -m)
if [ "$ARCH" != "arm64" ]; then
    echo -e "💡 Note: You are on $ARCH. MLX is optimized for Apple Silicon (arm64)."
fi

echo -e "${GREEN}✅ Environment looks good! You are ready to build mlx-rs.${NC}"
exit 0