
该项目用于测试cudnn对int8卷积计算的支持。

## 编译

修改makefile中CUDA和cudnn相关的变量，然后在当前路径下执行`make`，在`src`路径下的源文件会逐个被编译为独立的可执行文件并存在`./bin`路径下。

如果在执行可执行文件时，出现`error while loading shared libraries: libcudnn.so.*`，请将对应的库所在路径添加到环境变量`LD_LIBRARY_PATH`中。

## 测试内容


## cudnn int8卷积和float32卷积运行时间
width	40

height	40

channels	256	

batch size	16

out_channels	512	

kernel_size	3

gpu: P40

cuda8

cudnn6

迭代一万次 forward过程求平均

参数设置方式	随机数	

int8 1.69416ms

float32 6.618135ms


文件cudnn_conv_int8.cc	cudnn_conv_float32.cc		

