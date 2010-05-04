#!/bin/sh

rm cfrbm_movielen
make clean
make apex_tensor_cpu.o cpu=1
make apex_cfrbm.o cpu=1
make cfrbm_movielen cpu=1

echo "done"
