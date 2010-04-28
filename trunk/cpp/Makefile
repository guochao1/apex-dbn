# build apex tensor and rbm
ifeq ($(cpu), 1)
	GPU_MACRO:= -D__APEX_TENSOR_USE_GPU__=0 
else
	GPU_MACRO:= -D__APEX_TENSOR_USE_GPU__=1
endif

BIN:= srbm_model_drawer srbm_minist crbm_minist crbm_kyoto crbm_model_drawer crbm_model_dumper crbm_model_swaper kyoto_data_viewer crbm_filter_drawer crbm_model_modify
OPT:= -msse2 -O3 -fno-strict-aliasing -Wall $(GPU_MACRO)
PRJ_DIR:=..
TENSOR_DIR:=$(PRJ_DIR)/tensor
SRBM_DIR:=$(PRJ_DIR)/srbm
CRBM_DIR:=$(PRJ_DIR)/crbm
CFRBM_DIR:=$(PRJ_DIR)/cfrbm

ifeq ($(cpu), 1)
	NVCC :=g++
	OPTNVCC:=$(OPT) 
	TENSOR_LIB:=apex_tensor_cpu.o	
else
	NVCC :=nvcc
	OPTNVCC := -arch sm_11 -Xcompiler "$(OPT)"
	TENSOR_LIB:=apex_tensor_cpu.o apex_tensor_gpu.o			
endif

all: $(BIN)

clean:
	rm -f $(BIN) *.o *~ *.exe *.bmp

apex_srbm.o: $(SRBM_DIR)/apex_srbm.cpp 
	g++ $(OPT) -c -o apex_srbm.o  $(SRBM_DIR)/apex_srbm.cpp

apex_crbm.o: $(CRBM_DIR)/apex_crbm.cpp 
	g++ $(OPT) -c -o apex_crbm.o  $(CRBM_DIR)/apex_crbm.cpp

apex_cfrbm.o: $(CFRBM_DIR)/apex_cfrbm.cpp 
	g++ $(OPT) -c -o apex_cfrbm.o  $(CFRBM_DIR)/apex_cfrbm.cpp

cfrbm_movielen: cfrbm_movielen.cpp apex_cfrbm.o $(TENSOR_LIB)
	$(NVCC) $(OPTNVCC) -o cfrbm_movielen cfrbm_movielen.cpp apex_cfrbm.o $(TENSOR_LIB)

apex_tensor_cpu.o: $(TENSOR_DIR)/apex_tensor.h $(TENSOR_DIR)/apex_tensor_cpu.h $(TENSOR_DIR)/apex_tensor_cpu.cpp $(TENSOR_DIR)/apex_tensor_inline.cpp
	g++ $(OPT) -c -o apex_tensor_cpu.o  $(TENSOR_DIR)/apex_tensor_cpu.cpp 

apex_tensor_gpu.o: $(TENSOR_DIR)/apex_tensor.h $(TENSOR_DIR)/apex_tensor_gpu.h $(TENSOR_DIR)/apex_tensor_gpu.cu $(TENSOR_DIR)/apex_tensor_inline.cpp
	$(NVCC) $(OPTNVCC) -c -o apex_tensor_gpu.o $(TENSOR_DIR)/apex_tensor_gpu.cu

srbm_minist: srbm_minist.cpp apex_srbm.o $(TENSOR_LIB) 
	$(NVCC) $(OPTNVCC) -o srbm_minist srbm_minist.cpp apex_srbm.o $(TENSOR_LIB)

crbm_minist: crbm_minist.cpp apex_crbm.o $(TENSOR_LIB) 
	$(NVCC) $(OPTNVCC) -o crbm_minist crbm_minist.cpp  apex_crbm.o $(TENSOR_LIB) 

crbm_kyoto: crbm_kyoto.cpp apex_crbm.o $(TENSOR_LIB)  
	$(NVCC) $(OPTNVCC) -o crbm_kyoto crbm_kyoto.cpp apex_crbm.o $(TENSOR_LIB) 

srbm_model_drawer: srbm_model_drawer.cpp apex_tensor_cpu.o  
	g++ $(OPT) -lpthread -lX11  -o srbm_model_drawer srbm_model_drawer.cpp apex_tensor_cpu.o

crbm_model_drawer: crbm_model_drawer.cpp apex_tensor_cpu.o  
	g++ $(OPT) -lpthread -lX11  -o crbm_model_drawer crbm_model_drawer.cpp apex_tensor_cpu.o

crbm_filter_drawer: crbm_filter_drawer.cpp apex_tensor_cpu.o  
	g++ $(OPT) -lpthread -lX11  -o crbm_filter_drawer crbm_filter_drawer.cpp apex_tensor_cpu.o

crbm_model_dumper: crbm_model_dumper.cpp apex_tensor_cpu.o  
	g++ $(OPT) -o crbm_model_dumper crbm_model_dumper.cpp apex_tensor_cpu.o

crbm_model_swaper: crbm_model_swaper.cpp apex_tensor_cpu.o  
	g++ $(OPT) -o crbm_model_swaper crbm_model_swaper.cpp apex_tensor_cpu.o

crbm_model_modify: crbm_model_modify.cpp apex_tensor_cpu.o  
	g++ $(OPT) -o crbm_model_modify crbm_model_modify.cpp apex_tensor_cpu.o

kyoto_data_viewer: kyoto_data_viewer.cpp apex_tensor_cpu.o  
	g++ $(OPT) -lpthread -lX11  -o kyoto_data_viewer kyoto_data_viewer.cpp apex_tensor_cpu.o
