#!/bin/bash


jupyter nbconvert --to script keras_NeuralNetworks_solution.ipynb --output output

ipython grader.py

