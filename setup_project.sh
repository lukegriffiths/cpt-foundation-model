#!/bin/bash

# Create the main directories
echo "Creating directories..."
mkdir -p data/raw
mkdir -p notebooks
mkdir -p src

# Create the initial Python files and other text files
echo "Creating empty files..."
touch src/__init__.py
touch src/data_loader.py
touch src/model.py
touch src/train.py
touch notebooks/01_data_exploration.ipynb
touch requirements.txt

echo "Project structure created successfully!"