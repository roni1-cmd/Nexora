# Installation Guide

## Prerequisites

Make sure the following dependencies are installed on your system:

- **Python** 3.8+
- **CMake** 3.10+
- **Julia** 1.8+
- **Node.js** 18+
- **Maven** 3.8+
- **ONNX Runtime**
- **GCC/Clang** for C/C++ and Assembly

---

## Setup Instructions

### 1. Run the setup script
```bash
./scripts/setup.sh
```

### 2. Install Python dependencies
```bash
pip install -r requirements.txt
```

### 3. Install Node.js dependencies
```bash
cd src/js
npm install
```

### 4. Build C++ components
```bash
cd src/cpp/inference
mkdir build
cd build
cmake ..
make
```

---

You're now ready to start using the project!
