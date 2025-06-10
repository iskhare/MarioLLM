SETUP: 
1. Install latest drivers (570) from Nvidia along with driver utils (for nvidia-smi; not strictly necessary I think).
2. Download python (I'm using 3.12) and create a new venv and install requirements.
3. cd to llm_mario/ and run python main.py --train --save-screenshots
4. (Optional) set up wandb with `wandb login` and then run with --wandb

There will be some warnings about render mode and image/video processors, should be safe to ignore