
conda create --name EAsF python=3.10
conda activate EAsF

conda install pytorch==2.3.0 torchvision==0.18.0 torchaudio==2.3.0 pytorch-cuda=11.8 -c pytorch -c nvidia
conda install -c pyg pytorch-sparse
conda install -c pyg pytorch-scatter
conda install -c pyg pytorch-cluster
conda install pytorch-spline-conv -c pyg
conda install -c pyg pyg
pip install ogb pykeops
pip install matplotlib

