source /opt/mamba/init.bash
conda activate ml4co

python ../../common/evaluate.py ${@:1}
