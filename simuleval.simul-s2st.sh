export CUDA_VISIBLE_DEVICES=0
export SLURM_NTASKS=1
export PYTHONPATH=$PYTHONPATH:/gscratch/intelligentsystems/translation_groups/$USER/UW_StreamSpeech/fairseq # step1 change to your fairseq path (under StreamSpeech)
export PYTHONPATH=$PYTHONPATH:/gscratch/intelligentsystems/translation_groups/$USER/UW_StreamSpeech/SimulEval  # step2 change to your SimulEval path (under StreamSpeech)

ROOT=/gscratch/intelligentsystems/translation_groups/$USER/UW_StreamSpeech # step3: change to your StreamSpeech path
DATA_ROOT=/scr/data_streamspeech/cvss/cvss-c  # change to your data untar path
PRETRAIN_ROOT=$ROOT/pretrain_models 
VOCODER_CKPT=$PRETRAIN_ROOT/unit-based_HiFi-GAN_vocoder/mHuBERT.layer11.km1000.en/g_00500000
VOCODER_CFG=$PRETRAIN_ROOT/unit-based_HiFi-GAN_vocoder/mHuBERT.layer11.km1000.en/config.json

LANG=fr
file=/gscratch/intelligentsystems/translation_groups/$USER/simul-s2st.singlechannel-${LANG}/checkpoint_best.pt # step4: the checkpoint you trained 
output_dir=$ROOT/res/streamspeech.simultaneous.${LANG}-en/simul-s2st # step5: the output path to save output results and wavefiles

chunk_size=960

simuleval --data-bin ${DATA_ROOT}/${LANG}-en/fbank2unit \
    --user-dir ${ROOT}/researches/ctc_unity --agent-dir ${ROOT}/agent \
    --source ${DATA_ROOT}/${LANG}-en/simuleval/test/wav_list.txt --target  ${DATA_ROOT}/${LANG}-en/simuleval/test/target.txt \
    --model-path $file \
    --config-yaml config_gcmvn.yaml --multitask-config-yaml config_mtl_asr_st_ctcst.yaml \
    --agent $ROOT/agent/speech_to_speech.streamspeech.agent.py \
    --vocoder $VOCODER_CKPT --vocoder-cfg $VOCODER_CFG --dur-prediction \
    --output $output_dir/chunk_size=$chunk_size \
    --source-segment-size $chunk_size \
    --quality-metrics ASR_BLEU  --target-speech-lang en --latency-metrics AL AP DAL StartOffset EndOffset LAAL ATD NumChunks DiscontinuitySum DiscontinuityAve DiscontinuityNum RTF \
    --device gpu --computation-aware --start-index 0 --end-index 100 


# # To calculate ASR-BLEU w/o silence,
# # Another way: You can simply comment out Line 358 to Line 360 of StreamSpeech/SimulEval/simuleval/evaluator/instance.py to prevent silence from being added to the result within SimulEval.
#
# cd $ROOT/asr_bleu_rm_silence
# python compute_asr_bleu.py --reference_path ${DATA_ROOT}/${LANG}-en/simuleval/test/target.txt --lang en --audio_dirpath $output_dir/chunk_size=$chunk_size/wavs --reference_format txt --transcripts_path $output_dir/chunk_size=$chunk_size/rm_silence_asr_transcripts.txt --results_dirpath $output_dir/chunk_size=$chunk_size/rm_silence_asr_bleu
# cd $ROOT
