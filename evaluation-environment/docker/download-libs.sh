#!/bin/bash

set -e

if md5sum --check checksums.txt; then
    echo "==> Required libraries ok"
else
    echo "==> Downloading required libraries"
    rm -f cmake-3.20.2-linux-x86_64.sh
    rm -f v0.6.0.tar.gz
    rm -f SCIPOptSuite-7.0.2-Linux-debian.sh
    wget https://github.com/Kitware/CMake/releases/download/v3.20.2/cmake-3.20.2-linux-x86_64.sh
    wget https://github.com/ds4dm/ecole/archive/refs/tags/v0.6.0.tar.gz
    wget https://www.scipopt.org/download/release/SCIPOptSuite-7.0.2-Linux-debian.sh
    exec md5sum --check checksums.txt
    echo "==> Required libraries ok"
fi
