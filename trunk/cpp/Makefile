# Build apex tensor and rbm
ifeq ($(cpu), 1)
	GPU_MACRO:= -D__APEX_TENSOR_USE_GPU__=0 
else
	GPU_MACRO:= -D__APEX_TENSOR_USE_GPU__=1
endif

INSTALL_DIR := ../../bin

BIN:= crbm_light crbm_model_drawer crbm_model_dumper crbm_model_swaper kyoto_data_viewer crbm_filter_drawer crbm_model_modify crbm_kyoto_extractor 

LD_PATH:= -L/usr/local/cuda/lib
LD_OPT := $(LD_PATH) -lblas -lm -lcublas 
OPT:= -msse2 -O3 -fno-strict-aliasing -Wall $(GPU_MACRO)

PRJ_DIR:=..
TENSOR_DIR:=$(PRJ_DIR)/tensor
SRBM_DIR:=$(PRJ_DIR)/srbm
CRBM_DIR:=$(PRJ_DIR)/crbm/crbm_simple
CFRBM_DIR:=$(PRJ_DIR)/cfrbm

CC  := gcc
CPP := g++
CTENSOR_LIB:= dSFMT.o apex_tensor_cpu.o
	
ifeq ($(cpu), 1)
	NVCC := $(CPP)
	OPTNVCC:=$(OPT) 
	TENSOR_LIB:= $(CTENSOR_LIB)
else
	NVCC :=nvcc
	OPTNVCC := -arch sm_11 -Xcompiler "$(OPT)" 
	TENSOR_LIB:= apex_tensor_gpu.o $(CTENSOR_LIB)
endif

all: $(BIN)

install: 
	cp -f $(BIN) $(INSTALL_DIR)	

crbm: crbm_mnist crbm_kyoto crbm_model_drawer crbm_model_dumper crbm_model_swaper kyoto_data_viewer crbm_filter_drawer 
	
kdd : logistic_reg_factor logistic_reg_profile

clean:
	rm -f $(BIN) *.o *~ *.exe *.bmp

dSFMT.o: $(PRJ_DIR)/external/dSFMT/dSFMT.c
	$(CC) $(OPT) -c -o dSFMT.o $(PRJ_DIR)/external/dSFMT/dSFMT.c

apex_srbm.o  : $(SRBM_DIR)/apex_srbm.cpp 
	$(CPP) $(OPT) -c -o apex_srbm.o  $(SRBM_DIR)/apex_srbm.cpp

apex_crbm.o: $(CRBM_DIR)/apex_crbm.cpp 
	$(CPP) $(OPT) -c -o apex_crbm.o  $(CRBM_DIR)/apex_crbm.cpp

apex_crbm_light.o: $(PRJ_DIR)/crbm/crbm_light/apex_crbm.cpp 
	$(CPP) $(OPT) -c -o apex_crbm_light.o  $(PRJ_DIR)/crbm/crbm_light/apex_crbm.cpp 

apex_cfrbm.o: $(CFRBM_DIR)/apex_cfrbm.cpp 
	$(CPP) $(OPT) -c -o apex_cfrbm.o  $(CFRBM_DIR)/apex_cfrbm.cpp

cfrbm_movielen_train: cfrbm_movielen.cpp apex_cfrbm.o $(TENSOR_LIB)
	$(CPP) $(OPT) $(LD_OPT) -o cfrbm_movielen_train cfrbm_movielen.cpp apex_cfrbm.o $(TENSOR_LIB)

cfrbm_movielen_predict: cfrbm_movielen_predict.cpp $(TENSOR_LIB)
	$(CPP) $(OPT) $(LD_OPT) -o cfrbm_movielen_predict cfrbm_movielen_predict.cpp $(TENSOR_LIB)

cf_mf: cf_mf.cpp $(TENSOR_LIB)
	$(CPP) $(OPT) $(LD_OPT) -o cf_mf cf_mf.cpp $(TENSOR_LIB)

logistic_reg_factor: logistic_reg_factor.cpp $(CTENSOR_LIB)
	$(CPP) $(OPT) -o logistic_reg_factor logistic_reg_factor.cpp $(CTENSOR_LIB)

logistic_reg_profile: logistic_reg_profile.cpp $(CTENSOR_LIB)
	$(CPP) $(OPT) -o logistic_reg_profile logistic_reg_profile.cpp $(CTENSOR_LIB)

apex_tensor_cpu.o: $(TENSOR_DIR)/apex_tensor.h $(TENSOR_DIR)/apex_tensor_cpu.h $(TENSOR_DIR)/apex_tensor_cpu.cpp  $(TENSOR_DIR)/apex_tensor_cpu_inline.h $(TENSOR_DIR)/apex_tensor_inline.h
	$(CPP) $(OPT) -c -o apex_tensor_cpu.o $(TENSOR_DIR)/apex_tensor_cpu.cpp

apex_tensor_gpu.o: $(TENSOR_DIR)/apex_tensor.h $(TENSOR_DIR)/apex_tensor_gpu.h $(TENSOR_DIR)/apex_tensor_gpu.cu $(TENSOR_DIR)/apex_tensor_inline.h
	$(CPP) $(OPT) $(LD_OPT) -c -o apex_tensor_gpu.o $(TENSOR_DIR)/apex_tensor_gpu.cu

srbm_mnist: srbm_mnist.cpp apex_srbm.o $(TENSOR_LIB) 
	$(CPP) $(OPT) $(LD_OPT) -o srbm_mnist srbm_mnist.cpp apex_srbm.o $(TENSOR_LIB)

crbm_mnist: crbm_mnist.cpp apex_crbm.o $(TENSOR_LIB) 
	$(CPP) $(OPT) $(LD_OPT) -o crbm_mnist crbm_mnist.cpp  apex_crbm.o $(TENSOR_LIB) 

crbm_kyoto: crbm_kyoto.cpp apex_crbm.o $(TENSOR_LIB)  
	$(CPP) $(OPT) $(LD_OPT) -o crbm_kyoto crbm_kyoto.cpp apex_crbm.o $(TENSOR_LIB) 

crbm_light: crbm_light.cpp apex_crbm_light.o $(TENSOR_LIB)
	$(CPP) $(OPT) $(LD_OPT) -o crbm_light crbm_light.cpp apex_crbm_light.o $(TENSOR_LIB) 

crbm_kyoto_extractor: crbm_kyoto_extractor.cpp apex_crbm.o $(TENSOR_LIB)  
	$(CPP) $(OPT) $(LD_OPT) -o crbm_kyoto_extractor crbm_kyoto_extractor.cpp apex_crbm.o $(TENSOR_LIB) 

srbm_model_drawer: srbm_model_drawer.cpp $(CTENSOR_LIB)  
	$(CPP) $(OPT) $(LD_OPT) -lpthread -lX11  -o srbm_model_drawer srbm_model_drawer.cpp $(CTENSOR_LIB)

crbm_model_drawer: crbm_model_drawer.cpp $(CTENSOR_LIB)  
	$(CPP) $(OPT) $(LD_OPT) -lpthread -lX11  -o crbm_model_drawer crbm_model_drawer.cpp $(CTENSOR_LIB)

crbm_filter_drawer: crbm_filter_drawer.cpp $(CTENSOR_LIB)  
	$(CPP) $(OPT) $(LD_OPT) -lpthread -lX11  -o crbm_filter_drawer crbm_filter_drawer.cpp $(CTENSOR_LIB)

crbm_model_dumper: crbm_model_dumper.cpp $(CTENSOR_LIB)  
	$(CPP) $(OPT) $(LD_OPT) -o crbm_model_dumper crbm_model_dumper.cpp $(CTENSOR_LIB)

crbm_model_swaper: crbm_model_swaper.cpp $(CTENSOR_LIB)  
	$(CPP) $(OPT) $(LD_OPT) -o crbm_model_swaper crbm_model_swaper.cpp $(CTENSOR_LIB)

crbm_model_modify: crbm_model_modify.cpp $(CTENSOR_LIB)  
	$(CPP) $(OPT) $(LD_OPT) -o crbm_model_modify crbm_model_modify.cpp $(CTENSOR_LIB)

kyoto_data_viewer: kyoto_data_viewer.cpp $(CTENSOR_LIB)  
	$(CPP) $(OPT) $(LD_OPT) -lpthread -lX11 -I/usr/X11R6/include -o kyoto_data_viewer kyoto_data_viewer.cpp $(CTENSOR_LIB)
