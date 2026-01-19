
cd /root
#git clone -q https://github.com/munzuruleee/piper
git clone -q https://github.com/rmcpantoja/piper
cd /root/piper/src/python
wget -q "https://raw.githubusercontent.com/coqui-ai/TTS/dev/TTS/bin/resample.py"
pip install pip==24.0
pip install -q cython>=0.29.0 
pip install -q piper-phonemize-fix
pip install -q librosa>=0.9.2 
pip install -q numpy==1.24 
pip install -q onnxruntime>=1.11.0 
pip install -q pytorch-lightning==1.7.7 
pip install -q torch==1.13.0+cu117 --extra-index-url https://download.pytorch.org/whl/cu117
pip install -q torchtext==0.14.0 
pip install -q torchvision==0.14.0
pip install -q torchaudio==0.13.0 
pip install -q torchmetrics==0.11.4
pip install --upgrade gdown transformers
pip install matplotlib

pip install --upgrade torch torchvision torchaudio pytorch-lightning

bash build_monotonic_align.sh


mkdir /root/tts
mkdir /root/dataset
wget -P "/root/tts" "http://139.59.41.80/iitm_female_wav/wav.zip"
unzip "/root/tts/wav.zip" -d "/root/tts"
wget -P "/root/tts" "http://139.59.41.80/iitm_female_wav/metadata.csv"
mkdir /root/tts/wav
mkdir /root/audio_cache


export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export ONNXRUNTIME_NUM_THREADS=1


cd /root/tts
wget https://huggingface.co/datasets/rhasspy/piper-checkpoints/resolve/main/en/en_US/amy/medium/epoch%3D6679-step%3D1554200.ckpt

cd /root/piper/src/python/

rm -rf /root/tts/wavs_resampled

python resample.py --input_dir "/root/tts/wav" --output_dir "/root/tts/wavs_resampled" --output_sr 22050 --file_ext "wav"
mv /root/tts/wavs_resampled/* /root/tts/wav

echo "piper_train.preprocess..."



 


python -m piper_train.preprocess \
  --language bn \
  --input-dir /root/tts \
  --cache-dir "/root/audio_cache" \
  --output-dir "/root/piper/bangla_tts" \
  --dataset-name "bangla_tts" \
  --dataset-format ljspeech \
  --sample-rate 22050 \
  --single-speaker \
  --max-workers 1



echo "piper_train ..."


python -m piper_train \
  --dataset-dir "/root/piper/bangla_tts/" \
  --accelerator 'gpu' \
  --devices 1 \
  --batch-size 32 \
  --validation-split 0.01 \
  --num-test-examples 1 \
  --quality medium \
  --checkpoint-epochs 1 \
  --num_ckpt 2 \
  --save_last True \
  --log_every_n_steps 100 \
  --max_epochs 66790 \
  --precision 32 \
  --resume_from_checkpoint "/root/tts/epoch=6679-step=1554200.ckpt"



python -m piper_train \
  --dataset-dir "/root/piper/bangla_tts/" \
  --accelerator gpu \
  --devices 1 \
  --batch-size 32 \
  --validation-split 0.01 \
  --num-test-examples 1 \
  --quality medium \
  --checkpoint-epochs 1 \
  --log_every_n_steps 100 \
  --max_epochs 66790 \
  --precision 32 \
  --resume_from_checkpoint "/root/tts/epoch=6679-step=1554200.ckpt"


python3 -m piper.train.export_onnx \
  --checkpoint 1555028.ckpt \
  --output-file amymodel.onnx
