LOPT:= -lblas -lm -lpthread
ifeq ($(cpu), 1)
	GPU_MACRO:= -D__APEX_TENSOR_USE_GPU__=0 
else
	GPU_MACRO:= -D__APEX_TENSOR_USE_GPU__=1
endif

OPT := -msse2 -O3 -fno-strict-aliasing -Wall $(GPU_MACRO)
ifeq ($(cpu), 1)
	NVCC := g++
	OPTNVCC := $(OPT) $(LOPT)
	TENSOR_LIB:= apex_tensor_cpu.o dSFMT.o
else
	NVCC    :=nvcc		
	OPTNVCC := -arch sm_11 -Xcompiler "$(OPT)" -Xlinker "$(LOPT)" $(SHOW_PTX)
	TENSOR_LIB:= apex_tensor_cpu.o dSFMT.o apex_tensor_gpu.o	
endif
ifeq ($(showptx),1)
	SHOW_PTX:=--ptxas-options=-v
else
	SHOW_PTX:=
endif

BIN= test_cmp

PRJ_DIR=../..
TENSOR_DIR=$(PRJ_DIR)/tensor

CC  := gcc

all: $(BIN)

clean:
	rm -f $(BIN) *.o *~ *.exe

dSFMT.o: $(PRJ_DIR)/external/dSFMT/dSFMT.c
	$(CC) $(OPT) -c -o dSFMT.o $(PRJ_DIR)/external/dSFMT/dSFMT.c

apex_tensor_cpu.o: $(TENSOR_DIR)/apex_tensor.h $(TENSOR_DIR)/apex_tensor_cpu.h $(TENSOR_DIR)/apex_tensor_cpu.cpp $(TENSOR_DIR)/apex_tensor_inline.h
	g++ $(OPT) -c -o apex_tensor_cpu.o  $(TENSOR_DIR)/apex_tensor_cpu.cpp


apex_tensor_gpu.o: $(TENSOR_DIR)/apex_tensor.h $(TENSOR_DIR)/apex_tensor_gpu.h $(TENSOR_DIR)/apex_tensor_gpu.cu $(TENSOR_DIR)/apex_tensor_inline.h
	$(NVCC) $(OPTNVCC) -c -o apex_tensor_gpu.o $(TENSOR_DIR)/apex_tensor_gpu.cu

test_cmp: test_cmp.cpp test_cmp.h test_stats.h $(TENSOR_LIB)
	$(NVCC) $(OPTNVCC) -o test_cmp test_cmp.cpp $(TENSOR_LIB)

test_cmp_goto: test_cmp.cpp test_cmp.h test_stats.h ../../external/libgoto.a apex_tensor_cpu.o dSFMT.o 
	g++ $(OPT) -o test_cmp_goto test_cmp.cpp apex_tensor_cpu.o ../../external/libgoto.a  dSFMT.o -lpthread
