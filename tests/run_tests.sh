#!/bin/bash

echo "Beginning unit tests..."
pytest tests/ -v

echo "Running API notebook..."
pytest --nbmake docs/API_guide.ipynb

echo "Running coordinate examples notebook..."
pytest --nbmake examples/coordinate_systems_demo.ipynb