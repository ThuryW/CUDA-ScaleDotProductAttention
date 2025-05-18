### Progress of Week 1
1. 搭建项目仓库，熟悉CUDA编程流程
   - 仓库地址：https://github.com/ThuryW/CUDA-ScaleDotProductAttention
   - 后续代码更新都会放在这里
2. 初步学习CUDA编程，完成测试一个简单的cuBLAS矩阵乘法
   - 学习了CUDA编程基础，结合此前做过的C++项目基础开始着手开发
   - 花了比较多时间配环境，最初发现存在CUDA版本和gcc版本不兼容的问题（CUDA 10.0要求gcc版本不高于7）
   - 目前已经明确项目的编译流程和代码管理
     - `src`目录存放项目主程序
     - `include`目录存放项目头文件
     - `tests`目录存放测试模块代码
     - `data`目录用于存放输入数据
     - `scripts`目录用于存放脚本文件（如果后续要用的话）
     - `docs`目录存放项目相关文档
   - 项目采用CMake进行编译和管理，流程可参见`README.md`