#!/usr/bin/env bash
set -Eeuo pipefail

# Usage:
#   ./server_rebuild_and_test.sh [project_dir]
#
# Example:
#   ./server_rebuild_and_test.sh ~/CSC3060_Project5
#
# Run this ON THE SERVER.
# It will:
#   1) fetch origin/main
#   2) hard reset to origin/main
#   3) rebuild in build-server
#   4) run ./run_stu
#
# Warning:
# - This discards uncommitted changes in the server repo.

PROJECT_DIR="${1:-~/CSC3060_Project5}"

cd "${PROJECT_DIR}"
git fetch origin
git reset --hard origin/main

rm -rf build-server
cmake -S . -B build-server
cmake --build build-server -j

cd build-server
./run_stu | tee run_stu_latest.txt
cp run_stu_latest.txt ../run_stu_latest.txt

echo
echo "Saved result to: ${PROJECT_DIR}/run_stu_latest.txt"
