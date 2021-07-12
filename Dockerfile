# FROM continuumio/miniconda3
FROM ubuntu:20.04

ENV PROMPT_COMMAND='source /opt/mamba/init.bash; unset PROMPT_COMMAND'

# Minimum requirements for the image
RUN apt-get update
RUN apt-get install --yes --no-install-recommends wget git openssh-client tar gzip ca-certificates
RUN apt-get clean

COPY base.sh /workdir/base.sh
RUN bash /workdir/base.sh
