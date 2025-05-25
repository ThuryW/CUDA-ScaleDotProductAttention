# CUDA implement of ScaledDotProductAttention

## Build the project
1. Create a build directory:
```bash
mkdir build
cd build
```
2. Run CMake to configure the project:
```bash
cmake ..
# /home/wangtianyu/tools/cmake/bin/cmake .. # use specific version of cmake
```
3. Build the project:
```bash
make
```
4. Execute the program:
```bash
./main random_test # or other test program
```

## Verify the result
Currently, we use [Bert-base](https://github.com/google-research/bert) model parameters for test, which contains `12` attention heads, and the matrix size of each attention head is `128x64`.

1. Generate random matrice as input
```bash
python ../scripts/generate_qk.py
```
2. Run your executive program
3. Run the verification script
```bash
python ../scripts/verify_qk.py
```

## Requirements
Here is my testing environment:
1. g++ (GCC) 5.4.0
2. CUDA 10.0
3. cmake 3.30.0

You can make adjustments based on the conditions of your own device while ensuring compatibility