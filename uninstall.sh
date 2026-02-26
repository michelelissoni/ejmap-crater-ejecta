#!/bin/bash

python3 code/replace_paths.py code/ --replace $(pwd) __EJMAP_ROOT_DIR__ --dont-ask >> log/uninstall.log
python3 code/replace_paths.py data/ --recursive --replace $(pwd) __EJMAP_ROOT_DIR__ --dont-ask >> log/uninstall.log 
python3 code/replace_paths.py results/ --recursive --replace $(pwd) __EJMAP_ROOT_DIR__ --dont-ask >> log/uninstall.log
