# Baseline for the dual task

A Graph Neural Network model that chooses which variable to branch on. Training is done through imitation learning with Strong Branching as an expert. The training process has two steps: sample generation and training.

This directory takes the same form as a submission. As such, it contains the script to install all requirements following the evaluation pipeline with singularity (see init.sh).

For training on BENCHMARK:

1. Make sure instances are available on `../../instances`.

2. Generate samples
`python train_files/01_generate_dataset.py BENCHMARK`
Optional arguments:
`-s SEED`: random seed used to initialize the pseudo-random number generator
`-j NJOBS`: number of parallel sample-generation jobs.

3. Train on those samples
`python train_files/02_train.py BENCHMARK`
`-s SEED`: random seed used to initialize the pseudo-random number generator
`-g GPU`: CUDA GPU id (or -1 for CPU only)

When training, the file `train_files/trained_models/$BENCHMARK/best_params.pkl` will be generated. To evaluate the results copy the trained models into the `agents` directory, which imitates the final submission format. Follow the evaluation pipeline instructions to evaluate the generated parameters.
