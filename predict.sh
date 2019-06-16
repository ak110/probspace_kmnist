#!/bin/bash
set -eux
docker run --runtime=nvidia --interactive --tty --rm ak110/keras-docker ./averaging.py
