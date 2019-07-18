
该项目用于测试cudnn对int8卷积计算的支持。

## 编译

修改makefile中CUDA和cudnn相关的变量，然后在当前路径下执行`make`，在`src`路径下的源文件会逐个被编译为独立的可执行文件并存在`./bin`路径下。

如果在执行可执行文件时，出现`error while loading shared libraries: libcudnn.so.*`，请将对应的库所在路径添加到环境变量`LD_LIBRARY_PATH`中。

## 测试内容


