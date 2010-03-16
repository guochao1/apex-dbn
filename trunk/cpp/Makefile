BIN= srbm_minist crbm_minist
OPT= -msse2 -O3 -fno-strict-aliasing -Wall
PRJ_DIR=..
TENSOR_DIR=$(PRJ_DIR)/tensor
SRBM_DIR=$(PRJ_DIR)/srbm
CRBM_DIR=$(PRJ_DIR)/crbm
all: $(BIN)

clean:
	rm -f $(BIN) *.o *~ *.exe

apex_srbm.o: $(SRBM_DIR)/apex_srbm.cpp 
	g++ $(OPT) -c -o apex_srbm.o  $(SRBM_DIR)/apex_srbm.cpp

apex_crbm.o: $(CRBM_DIR)/apex_crbm.cpp 
	g++ $(OPT) -c -o apex_crbm.o  $(CRBM_DIR)/apex_crbm.cpp

apex_tensor_cpu.o: $(TENSOR_DIR)/apex_tensor.h $(TENSOR_DIR)/apex_tensor_cpu.h $(TENSOR_DIR)/apex_tensor_cpu.cpp $(TENSOR_DIR)/apex_tensor_inline.cpp
	g++ $(OPT) -c -o apex_tensor_cpu.o  $(TENSOR_DIR)/apex_tensor_cpu.cpp

srbm_minist: srbm_minist.cpp apex_tensor_cpu.o  apex_srbm.o
	g++ $(OPT) -o srbm_minist srbm_minist.cpp apex_tensor_cpu.o apex_srbm.o 

crbm_minist: crbm_minist.cpp apex_tensor_cpu.o  apex_crbm.o
	g++ $(OPT) -o crbm_minist crbm_minist.cpp apex_tensor_cpu.o apex_crbm.o 

srbm_model_drawer: srbm_model_drawer.cpp apex_tensor_cpu.o  
	g++ $(OPT) -o srbm_model_drawer srbm_model_drawer.cpp apex_tensor_cpu.o