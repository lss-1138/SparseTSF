#!/bin/bash

# linear
sh scripts/SparseTSF/linear/etth1.sh;
sh scripts/SparseTSF/linear/etth2.sh;
sh scripts/SparseTSF/linear/electricity.sh;
sh scripts/SparseTSF/linear/traffic.sh;
sh scripts/SparseTSF/linear/ettm1.sh;
sh scripts/SparseTSF/linear/ettm2.sh;
sh scripts/SparseTSF/linear/weather.sh;
sh scripts/SparseTSF/linear/solar.sh;


# mlp
sh scripts/SparseTSF/mlp/etth1.sh;
sh scripts/SparseTSF/mlp/etth2.sh;
sh scripts/SparseTSF/mlp/electricity.sh;
sh scripts/SparseTSF/mlp/traffic.sh;
sh scripts/SparseTSF/mlp/ettm1.sh;
sh scripts/SparseTSF/mlp/ettm2.sh;
sh scripts/SparseTSF/mlp/weather.sh;
sh scripts/SparseTSF/mlp/solar.sh;



