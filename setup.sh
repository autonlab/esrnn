# ts_transfer packages
conda create --name d3m_esrnn python=3.6
source ~/anaconda3/etc/profile.d/conda.sh
conda activate d3m_esrnn

# basic
conda install -c anaconda numpy==1.16.1
conda install -c anaconda pandas==0.25.2

# visualization
conda install -c conda-forge matplotlib==3.1.1

# pytorch
conda install pytorch=1.3.1 -c pytorch

conda install -c conda-forge jupyterlab
ipython kernel install --user --name=d3m_esrnn
conda deactivate
