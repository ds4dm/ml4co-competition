##### System definition #####

FROM python:3.9-slim

# Required build tools to work with cmake.
RUN apt-get update \
 && apt-get install -y make gcc g++ gfortran liblapack3 libtbb2 libcliquer1 libopenblas-dev libgsl23

# Need to get a later version of cmake installed than than available in buster.
# Source: https://github.com/Kitware/CMake/releases/download/v3.20.2/cmake-3.20.2-linux-x86_64.sh
# Source: https://cmake.org/download/
COPY cmake-3.20.2-linux-x86_64.sh cmake.sh
RUN sh cmake.sh --skip-license
RUN rm cmake.sh

##### Virtualenv builder #####

WORKDIR /ecole
# Latest SCIP release
# Source: https://www.scipopt.org/download/release/SCIPOptSuite-7.0.2-Linux-debian.sh
# Source: https://www.scipopt.org/download.php?fname=SCIPOptSuite-7.0.2-Linux-debian.sh
COPY SCIPOptSuite-7.0.2-Linux-debian.sh SCIPOptSuite-7.0.2-Linux-debian.sh
# Using a source release version of ecole.
# Avoiding conda since I assume we'll use something customised?
# Source: https://github.com/ds4dm/ecole/archive/refs/tags/v0.6.0.tar.gz
# Source: https://github.com/ds4dm/ecole/releases/tag/v0.6.0
COPY v0.6.0.tar.gz v0.6.0.tar.gz

# Copy and run venv build script
COPY build-venv-ecole.sh build-venv-ecole.sh
RUN bash build-venv-ecole.sh

##### Test run #####

# Placeholder - just run a test script for now, users code to be built-in?
COPY ecole-test.py ecole-test.py
CMD python ecole-test.py
