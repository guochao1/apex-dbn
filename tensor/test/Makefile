ifeq ($(cpu), 1)
	GPU_MACRO:= -D__APEX_TENSOR_USE_GPU__=0 
else
	GPU_MACRO:= -D__APEX_TENSOR_USE_GPU__=1
endif
ifeq ($(showptx),1)
	SHOW_PTX:=--ptxas-options=-v
else
	SHOW_PTX:=
endif

BIN= test_cmp
OPT= -msse2 -O3 -fno-strict-aliasing -Wall $(GPU_MACRO)
PRJ_DIR=../..
TENSOR_DIR=$(PRJ_DIR)/tensor
NVCC    =nvcc
OPTNVCC = -arch sm_11 -Xcompiler "$(OPT)" $(SHOW_PTX)

all: $(BIN)

clean:
	rm -f $(BIN) *.o *~ *.exe

apex_tensor_cpu.o: $(TENSOR_DIR)/apex_tensor.h $(TENSOR_DIR)/apex_tensor_cpu.h $(TENSOR_DIR)/apex_tensor_cpu.cpp $(TENSOR_DIR)/apex_tensor_inline.cpp
	g++ $(OPT) -c -o apex_tensor_cpu.o  $(TENSOR_DIR)/apex_tensor_cpu.cpp

apex_tensor_xpu.o: $(TENSOR_DIR)/apex_tensor_xpu_inline.cpp
	g++ $(OPT) -c -o apex_tensor_xpu.o  $(TENSOR_DIR)/apex_tensor_xpu_inline.cpp	

apex_tensor_gpu.o: $(TENSOR_DIR)/apex_tensor.h $(TENSOR_DIR)/apex_tensor_gpu.h $(TENSOR_DIR)/apex_tensor_gpu.cu $(TENSOR_DIR)/apex_tensor_inline.cpp
	$(NVCC) $(OPTNVCC) -c -o apex_tensor_gpu.o $(TENSOR_DIR)/apex_tensor_gpu.cu

test_cmp: test_cmp.cpp test_cmp.h test_stats.h apex_tensor_cpu.o apex_tensor_gpu.o  apex_tensor_xpu.o
	$(NVCC) $(OPTNVCC) -o test_cmp test_cmp.cpp apex_tensor_cpu.o apex_tensor_gpu.o  
