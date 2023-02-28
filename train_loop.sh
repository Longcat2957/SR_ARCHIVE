#!/bin/sh

echo "Start train loop"
python train.py --model ABPN --e 600 --complex_da --wobn
python train.py --model ABPNV2 --e 600 --complex_da --wobn