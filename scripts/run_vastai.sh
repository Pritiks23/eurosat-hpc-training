# Vast.ai run script placeholder
#!/bin/bash

set -e


echo "Starting EuroSAT GPU Training"


# Activate environment if available

if [ -d ".venv" ]

then

    source .venv/bin/activate

fi



echo "GPU Information"

nvidia-smi



echo ""
echo "Starting CUDA benchmark"
echo ""


python benchmark.py



echo ""
echo "Starting Training"
echo ""


python train.py



echo ""
echo "Running Evaluation"
echo ""


python evaluate.py



echo ""
echo "Generating Predictions"
echo ""


python visualize.py



echo ""
echo "Pipeline Complete"

