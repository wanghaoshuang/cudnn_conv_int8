
CUDA_INCLUDE=/home/work/cuda-9.0/include/
CUDA_RT_LIBS=/home/work/cuda-9.0/lib64/
CUDA_LIBS=/home/work/cuda-9.0/lib64/stubs/
CUDNN_INCLUDE=/home/work/cudnn/cudnn_v7/cuda/include/
CUDNN_LIBS=/home/work/cudnn/cudnn_v7/cuda/lib64/


CC = g++
CFLAGS = -std=c++11 -o test -lcudart -lrt -lcuda -lcudnn
INCLUDE = -I$(CUDA_INCLUDE) \
          -I$(CUDNN_INCLUDE)
LIBRARY = -L$(CUDA_LIBS) \
          -L$(CUDNN_LIBS) \
          -L$(CUDA_RT_LIBS)
SRC_DIR = ./src
SRCS = $(wildcard $(SRC_DIR)/*.cc)
targets = $(foreach src, $(SRCS), $(basename $(src)).a)
$(info $(SRCS))
$(info $(targets))

all: $(targets)

$(SRC_DIR)/%.a: $(SRC_DIR)/%.cc
	$(CC) $(CFLAGS) $(INCLUDE) $(LIBRARY) -o $@ $<  


.PHONY: all clean
clean:
	rm -rf $(SRC_DIR)/*.a
