#!/bin/bash

# Script that fetches all necessary data for training and eval

# Model constants etc.
wget http://visiondata.cis.upenn.edu/spin/data.tar.gz && tar -xvf data.tar.gz && rm data.tar.gz
# Initial fits to start training
wget http://visiondata.cis.upenn.edu/spin/static_fits.tar.gz && tar -xvf static_fits.tar.gz --directory data && rm -r static_fits.tar.gz
# List of preprocessed .npz files for each dataset
wget http://visiondata.cis.upenn.edu/spin/dataset_extras.tar.gz && tar -xvf dataset_extras.tar.gz --directory data && rm -r dataset_extras.tar.gz
# Pretrained checkpoint
wget http://visiondata.cis.upenn.edu/spin/model_checkpoint.pt --directory-prefix=data

# GMM prior from vchoutas/smplify0x
wget https://github.com/vchoutas/smplify-x/raw/master/smplifyx/prior.py -O smplify/prior.py

# Get the .npz files that for datasets
wget --no-check-certificate 'https://docs.google.com/uc?export=download&id=1-Pb3cGy_wsNR6ypbn_oD5FRoqRMDJ9xm' -O data/dataset_extras/3doh50k_test.npz
wget --no-check-certificate 'https://docs.google.com/uc?export=download&id=1-dDfH5ykGLibu5_0Q-sqaa6uoS9gwjDr' -O data/dataset_extras/3dpw_train_m.npz
wget --no-check-certificate 'https://docs.google.com/uc?export=download&id=1-gHCFh3D0HM1k_0FJXQXro7jeCSfUCCW' -O data/dataset_extras/h36m_valid_protocol1_m.npz
wget --no-check-certificate 'https://docs.google.com/uc?export=download&id=1-TZwSPIAbgg1Yf_Oel2V58cDeI2_RDTS' -O data/dataset_extras/3dpw_test_m.npz