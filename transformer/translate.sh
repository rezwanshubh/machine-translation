#!/usr/bin/env bash
python src_enc.py

python ./OpenNMT-py/translate.py -model ./models/_step_350000.pt -src ./en.enc.txt -output ./et.enc.txt -replace_unk -verbose

python tgt_dec.py