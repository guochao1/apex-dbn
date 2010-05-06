#!/bin/sh

rm cfrbm_movielen_train
make clean
make apex_tensor_cpu.o cpu=1
make apex_cfrbm.o cpu=1
make cfrbm_movielen_train cpu=1

echo "done"
