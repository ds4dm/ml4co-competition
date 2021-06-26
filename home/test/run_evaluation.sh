source /opt/mamba/init.bash
conda activate ml4co

python tasks/evaluate.py ${@:1}
