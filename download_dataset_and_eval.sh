#!/bin/bash

tar -xvf /gscratch/intelligentsystems/common_datasets/translation/data_stream_1channel_processed.tar.gz -C /scr/

source .venv/bin/activate

./simuleval.simul-s2st.sh
