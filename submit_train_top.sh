#!/bin/bash

# Define the parameters
params=(
  "cfg0 eb"
)

# Loop over the parameters
for param in "${params[@]}"; do
  IFS=' ' read -r -a param_array <<< "$param"
  param1=${param_array[0]}
  param2=${param_array[1]}

  # Create a batch script for this set of parameters
  batch_script=$(sed "s/{param1}/$param1/g; s/{param2}/$param2/g" jobs/train_top.sub)

  # Submit the batch script to Slurm
  echo "$batch_script" | sbatch
done
