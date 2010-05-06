#!/bin/sh

rm cfrbm_movielen_predict
make clean
make cfrbm_movielen_predict cpu=1

echo "done"
