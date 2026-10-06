#!/bin/bash
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=1
export KMP_DUPLICATE_LIB_OK=TRUE
export PYTORCH_ENABLE_MPS_FALLBACK=1

streamlit run app.py --server.fileWatcherType none
