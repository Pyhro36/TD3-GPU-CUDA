#!/bin/bash

rsync -e ssh -avz --exclude-from=".gitignore" . plefebvre@access.grid5000.fr:lille/INSA-5IF/Sema31/TD3-GPU-CUDA
