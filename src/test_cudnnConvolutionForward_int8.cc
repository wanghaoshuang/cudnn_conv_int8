#include <algorithm>
#include <memory>
#include <string>
#include <vector>
#include <cmath>
#include <stdint.h>   // for the typedefs (redundant, actually)
#include <inttypes.h> //
#include <ctime>
#include <cfloat>
#include <iomanip>
#include <iostream>
#include <map>
#include <random>
#include <sstream>

#include "cuda.h"
#include "cudnn.h"
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#define DATA_TYPE int8_t
#define OUT_DATA_TYPE float
#define CUDNN_DATA_TYPE CUDNN_DATA_INT8
#define CUDNN_OUT_TYPE CUDNN_DATA_FLOAT
#define CUDNN_COMPUTE_TYPE CUDNN_DATA_INT32

/** Error handling from https://developer.nvidia.com/cuDNN */
#define FatalError(s)                                                          \
  do {                                                                         \
    std::stringstream _where, _message;                                        \
    _where << __FILE__ << ':' << __LINE__;                                     \
    _message << std::string(s) + "\n" << __FILE__ << ':' << __LINE__;          \
    std::cerr << _message.str() << "\nAborting...\n";                          \
    cudaDeviceReset();                                                         \
    exit(1);                                                                   \
  } while (0)

#define checkCUDNN(status)                                                     \
  do {                                                                         \
    std::stringstream _error;                                                  \
    if (status != CUDNN_STATUS_SUCCESS) {                                      \
      _error << "CUDNN failure: " << cudnnGetErrorString(status);              \
      FatalError(_error.str());                                                \
    }                                                                          \
  } while (0)

#define checkCudaErrors(status)                                                \
  do {                                                                         \
    std::stringstream _error;                                                  \
    if (status != 0) {                                                         \
      _error << "Cuda failure: " << status;                                    \
      FatalError(_error.str());                                                \
    }                                                                          \
  } while (0)

/** Convolutional layer */
struct ConvolutionLayer {
  int kernel_size;
  int in_channels, in_height, in_width;
  int out_channels, out_height, out_width;
  DATA_TYPE* pconv;
  size_t pconv_size;

  ConvolutionLayer(int in_channels_,
                   int out_channels_,
                   int kernel_size_,
                   int in_w_,
                   int in_h_) {
    pconv_size = in_channels_ * kernel_size_ * kernel_size_ * out_channels_;
    in_channels = in_channels_;
    out_channels = out_channels_;
    kernel_size = kernel_size_;
    in_width = in_w_;
    in_height = in_h_;
    out_width = in_w_ - kernel_size_ + 1;
    out_height = in_h_ - kernel_size_ + 1;
  }
};

/** Training context */
struct TrainingContext {
  cudnnHandle_t cudnnHandle;
  cudnnTensorDescriptor_t dataTensor, convTensor, convBiasTensor;
  cudnnFilterDescriptor_t convFilterDesc;
  cudnnConvolutionDescriptor_t convDesc;
  cudnnConvolutionFwdAlgo_t convAlgo;
  int m_gpuid;
  int m_batchSize;
  size_t m_workspaceSize;

  // Disable copying
  TrainingContext& operator=(const TrainingContext&) = delete;
  TrainingContext(const TrainingContext&) = delete;

  // Constructor
  TrainingContext(int gpuid, int batch_size, ConvolutionLayer& conv)
    : m_gpuid(gpuid) {
    m_batchSize = batch_size;

    /** Create descriptors within the constructor.
      * As instructed in the Usual manual, descriptors for
      * input and output tensors, filter, and the forward
      * convolution operator are created along with
      * cuDNN handle.
      */
    checkCudaErrors(cudaSetDevice(gpuid));
    checkCUDNN(cudnnCreate(&cudnnHandle));
    checkCUDNN(cudnnCreateTensorDescriptor(&dataTensor));
    checkCUDNN(cudnnCreateFilterDescriptor(&convFilterDesc));
    checkCUDNN(cudnnCreateConvolutionDescriptor(&convDesc));
    checkCUDNN(cudnnCreateTensorDescriptor(&convTensor));

    // Initialize convolution forward pass
    size_t workspaceSizeFromConv = SetFwdConvolutionTensors(
        conv, dataTensor, convTensor, convFilterDesc, convDesc, convAlgo);
    m_workspaceSize = std::max((int)workspaceSizeFromConv, 0);
  }

  ~TrainingContext() {
    checkCudaErrors(cudaSetDevice(m_gpuid));
    checkCUDNN(cudnnDestroy(cudnnHandle));

    checkCUDNN(cudnnDestroyTensorDescriptor(dataTensor));
    checkCUDNN(cudnnDestroyTensorDescriptor(convTensor));
    checkCUDNN(cudnnDestroyFilterDescriptor(convFilterDesc));
    checkCUDNN(cudnnDestroyConvolutionDescriptor(convDesc));
  }

  /** Set tensors and ops for forward pass */
  size_t SetFwdConvolutionTensors(ConvolutionLayer& conv,
                                  cudnnTensorDescriptor_t& srcTensorDesc,
                                  cudnnTensorDescriptor_t& dstTensorDesc,
                                  cudnnFilterDescriptor_t& filterDesc,
                                  cudnnConvolutionDescriptor_t& convDesc,
                                  cudnnConvolutionFwdAlgo_t& algo) {
    int n = m_batchSize;
    int c = conv.in_channels;
    int h = conv.in_height;
    int w = conv.in_width;
    printf("xDesc: n=%d; c=%d; h=%d; w=%d\n",n,c,h,w);
    // Set input tensor. Folowing the manual, chagnged
    // * CUDNN_DATA_FLOAT -> CUDNN_DATA_INT8, and 
    // * CUDNN_TENSOR_NCHW -> CUDNN_TENSOR_NHWC
    checkCUDNN(cudnnSetTensor4dDescriptor(
        srcTensorDesc, CUDNN_TENSOR_NHWC, CUDNN_DATA_TYPE, n, c, h, w));

    // Set convolution filter. Folowing the manual, chagnged
    // * CUDNN_DATA_FLOAT -> CUDNN_DATA_INT8, and 
    // * CUDNN_TENSOR_NCHW -> CUDNN_TENSOR_NHWC
    checkCUDNN(cudnnSetFilter4dDescriptor(filterDesc,
                                          CUDNN_DATA_TYPE,
                                          CUDNN_TENSOR_NHWC,
                                          conv.out_channels,
                                          conv.in_channels,
                                          conv.kernel_size,
                                          conv.kernel_size));

    // Set convolution operator. Folowing the manual, chagnged
    // * CUDNN_DATA_FLOAT -> CUDNN_DATA_INT32
    int pad_height = 0;
    int pad_width = 0;
    int stride_h = 1;
    int stride_v = 1;
    int dilation_h = 1;
    int dilation_w = 1;
    checkCUDNN(cudnnSetConvolution2dDescriptor(convDesc,
                                               pad_height,
                                               pad_width,
                                               stride_h,
                                               stride_v,
                                               dilation_h,
                                               dilation_w,
                                               CUDNN_CONVOLUTION,
                                               CUDNN_COMPUTE_TYPE));

    // Compute output dimension. Folowing the manual, chagnged
    // * CUDNN_DATA_FLOAT -> CUDNN_DATA_INT8, and 
    // * CUDNN_TENSOR_NCHW -> CUDNN_TENSOR_NHWC
    checkCUDNN(cudnnGetConvolution2dForwardOutputDim(
        convDesc, srcTensorDesc, filterDesc, &n, &c, &h, &w));

    printf("yDesc: n=%d; c=%d; h=%d; w=%d\n",n,c,h,w);
    // Set output tensor (Changed CUDNN_DATA_FLOAT to CUDNN_DATA_INT8, following the manual)
    checkCUDNN(cudnnSetTensor4dDescriptor(
        dstTensorDesc, CUDNN_TENSOR_NHWC, CUDNN_OUT_TYPE, n, c, h, w));

    // Retrieve orward pass algorithm. We can either hardcode it to a specific
    // algorithm or use cudnnGetConvolutionForwardAlgorithm. For the purpose
    // of this test, either way works.
    algo = CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_PRECOMP_GEMM;
    // Following also works
//     if ((cudnnHandle == NULL) or (srcTensorDesc==NULL) or (filterDesc==NULL) or (convDesc == NULL) or (dstTensorDesc==NULL)) {
//       printf("desc NULL");
//     }
//     checkCUDNN(cudnnGetConvolutionForwardAlgorithm(
//         cudnnHandle,
//         srcTensorDesc,
//         filterDesc,
//         convDesc,
//         dstTensorDesc,
//         CUDNN_CONVOLUTION_FWD_PREFER_FASTEST,
//         0,
//         &algo));
//    

    // Compute workspace size. We can either hardcode it to a specific number,
    // or use cudnnGetConvolutionForwardWorkspaceSize. For the purpose of this
    // test, either way works.
    size_t sizeInBytes = 1073741824;    
    // Following also works
    // size_t sizeInBytes = 0;
    // checkCUDNN(cudnnGetConvolutionForwardWorkspaceSize(cudnnHandle,
    //                                                    srcTensorDesc,
    //                                                    filterDesc,
    //                                                    convDesc,
    //                                                    dstTensorDesc,
    //                                                    algo,
    //                                                    &sizeInBytes));
    

    return sizeInBytes;
  }

  /** Execute forward pass */
  void ForwardPropagation(DATA_TYPE* data,
                          OUT_DATA_TYPE* conv,
                          DATA_TYPE* pconv,
                          void* workspace) {
    float alpha = 1.0f;
    float beta = 0.0f;
    checkCudaErrors(cudaSetDevice(m_gpuid));
    checkCUDNN(cudnnConvolutionForward(cudnnHandle,
                                       &alpha,
                                       dataTensor,
                                       data,
                                       convFilterDesc,
                                       pconv,
                                       convDesc,
                                       convAlgo,
                                       workspace,
                                       m_workspaceSize,
                                       &beta,
                                       convTensor,
                                       conv));
  }
};

int main() {
  // parameters
  int gpu = 0;
  int iterations = 10000;

  // input dimensions
  size_t width = 4;
  size_t height = 4;
  size_t channels = 4;
  int batch_size = 1;

  // Create layer architecture
  int out_channels = 4;
  int kernel_size = 2;
  ConvolutionLayer conv(
      (int)channels, out_channels, kernel_size, (int)width, (int)height);
  TrainingContext context(gpu, batch_size, conv);

  printf("context.m_batchSize: %d; conv.out_channels: %d; conv.out_height: %d; conv.out_width: %d", context.m_batchSize, conv.out_channels, conv.out_height, conv.out_width);

  // weights 
  size_t filter_data_size = channels * kernel_size * kernel_size * out_channels;
  DATA_TYPE* filter_host_data = (DATA_TYPE*)malloc(sizeof(DATA_TYPE) * filter_data_size);
  for (size_t i=0; i< filter_data_size; i++) {
    filter_host_data[i] = static_cast<DATA_TYPE>(22);
  }

  // image
  size_t img_data_size = 1 * width * height * channels;
  DATA_TYPE* img_host_data = (DATA_TYPE*)malloc(sizeof(DATA_TYPE) * img_data_size);
  for (int i=0; i< img_data_size; i++) {
    img_host_data[i] = static_cast<DATA_TYPE>(1);
  }


  // output
  size_t out_data_size = context.m_batchSize * conv.out_channels * conv.out_height * conv.out_width;
  OUT_DATA_TYPE* out_host_data = (OUT_DATA_TYPE*)malloc(sizeof(OUT_DATA_TYPE) * out_data_size);
  for (int i=0; i< out_data_size; i++) {
    out_host_data[i] = static_cast<OUT_DATA_TYPE>(0);
  }


  DATA_TYPE* img_device_data;
  checkCudaErrors(cudaMalloc(&img_device_data,
                             sizeof(DATA_TYPE) * img_data_size));
  OUT_DATA_TYPE* out_device_data;
  checkCudaErrors(cudaMalloc(&out_device_data,
                             sizeof(OUT_DATA_TYPE) * out_data_size));

  DATA_TYPE* filter_device_data;
  checkCudaErrors(cudaMalloc(&filter_device_data, sizeof(DATA_TYPE) * filter_data_size));


  checkCudaErrors(cudaMemcpy(img_device_data,
                                  img_host_data,
                                  sizeof(DATA_TYPE) * img_data_size,
                                  cudaMemcpyHostToDevice));

  checkCudaErrors(cudaMemcpy(out_device_data,
                                  out_host_data,
                                  sizeof(OUT_DATA_TYPE) * out_data_size,
                                  cudaMemcpyHostToDevice));

  checkCudaErrors(cudaMemcpy(filter_device_data,
                                  filter_host_data,
                                  sizeof(DATA_TYPE) * filter_data_size,
                                  cudaMemcpyHostToDevice));


  // Temporary buffers and workspaces
  void* d_cudnn_workspace = nullptr;
  if (context.m_workspaceSize > 0) {
    checkCudaErrors(cudaMalloc(&d_cudnn_workspace, context.m_workspaceSize));
  }

  // Start forward pass
  printf("Begin forwrad pass\n");
  checkCudaErrors(cudaDeviceSynchronize());
  context.ForwardPropagation(img_device_data, out_device_data, filter_device_data, d_cudnn_workspace);
  checkCudaErrors(cudaDeviceSynchronize());

  cudaMemcpy(img_host_data,
          img_device_data,
          sizeof(DATA_TYPE) * img_data_size,
          cudaMemcpyDeviceToHost);
  checkCudaErrors(cudaDeviceSynchronize());
  printf("img data: \n");
  for(int i=0; i<img_data_size; i++) {
      printf("%"PRIi8"\t", img_host_data[i]);
  }
  printf("\n");

  cudaMemcpy(filter_host_data,
          filter_device_data,
          sizeof(DATA_TYPE) * filter_data_size,
          cudaMemcpyDeviceToHost);
  checkCudaErrors(cudaDeviceSynchronize());
  printf("filter data: \n");
  for(int i=0; i<filter_data_size; i++) {
      printf("%"PRIi8"\t", filter_host_data[i]);
  }
  printf("\n");

  printf("\nout_data_size: %d\n", out_data_size);
  cudaMemcpy(out_host_data,
          out_device_data,
          sizeof(OUT_DATA_TYPE) * out_data_size,
          cudaMemcpyDeviceToHost);
  checkCudaErrors(cudaDeviceSynchronize());
  printf("out data: \n");
  for(int i=0; i<out_data_size; i++) {
      // printf("%"PRIi8"\t", out_host_data[i]);
      printf("%f\t", out_host_data[i]);
  }
  printf("\n");
  
  

  // Free data structures
  checkCudaErrors(cudaFree(img_device_data));
  checkCudaErrors(cudaFree(out_device_data));
  checkCudaErrors(cudaFree(filter_device_data));

  if (d_cudnn_workspace != nullptr)
    checkCudaErrors(cudaFree(d_cudnn_workspace));

  return 0;
}
