
CUDA_INCLUDE=/home/work/cuda-8.0/include/
CUDA_RT_LIBS=/home/work/cuda-8.0/lib64/
CUDA_LIBS=/home/work/cuda-8.0/lib64/stubs/
CUDNN_INCLUDE=/home/work/cudnn/cudnn_v6/cuda/include/
CUDNN_LIBS=/home/work/cudnn/cudnn_v6/cuda/lib64/
SRC_DIR = ./src
BIN_DIR = ./bin


CC = g++
CFLAGS = -std=c++11 -o test -lcudart -lrt -lcuda -lcudnn
INCLUDE = -I$(CUDA_INCLUDE) \
          -I$(CUDNN_INCLUDE)
LIBRARY = -L$(CUDA_LIBS) \
          -L$(CUDNN_LIBS) \
          -L$(CUDA_RT_LIBS)
SRCS = $(wildcard $(SRC_DIR)/*.cc)
targets = $(foreach src, $(SRCS), $(basename $(src)).a)
$(info $(SRCS))
$(info $(targets))

all: $(targets)

$(SRC_DIR)/%.a: $(SRC_DIR)/%.cc
	$(CC) $(CFLAGS) $(INCLUDE) $(LIBRARY) -o $@ $<  
	mv -f $@ ./$(BIN_DIR)/

.PHONY: all clean
clean:
	rm -rf $(BIN_DIR)/*.a
	rm -rf $(SRC_DIR)/*.a
