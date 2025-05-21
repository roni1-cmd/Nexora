#!/bin/bash

# Create directory structure
mkdir -p data/{raw,processed,output/{predictions,models,metrics}}
mkdir -p logs
mkdir -p src/{python/{models,utils},cpp/inference,julia,c,js/api/routes,java/src/main/java/nexora,asm}
mkdir -p tests/{python,cpp,julia,c,js,java/src/test/java/nexora}
mkdir -p docs configs

# Install Python dependencies
pip install -r requirements.txt

echo "Environment setup complete"
