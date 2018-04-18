#!/usr/bin/env bash

echo "build solvers..."
python bbox_setup.py build_ext --inplace

echo "build psroi_pooling..."
cd models/psroi_pooling
sh make.sh
