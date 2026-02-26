#!/bin/bash

python3 code/replace_paths.py code/ --replace __EJMAP_ROOT_DIR__ $(pwd) --dont-ask >> log/install.log
python3 code/replace_paths.py data/ --recursive --replace __EJMAP_ROOT_DIR__ $(pwd) --dont-ask >> log/install.log 
python3 code/replace_paths.py results/ --recursive --replace __EJMAP_ROOT_DIR__ $(pwd) --dont-ask >> log/install.log
