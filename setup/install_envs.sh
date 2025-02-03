pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

cd src/env/dm_control
pip install -e .

cd ../dmc2gym
pip install -e .

cd ../../..
