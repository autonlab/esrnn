# ts_transfer packages
conda create --name d3m_esrnn python=3.6
source ~/anaconda3/etc/profile.d/conda.sh
conda activate d3m_esrnn

# basic
conda install -c anaconda numpy==1.16.1
conda install -c anaconda pandas==0.25.2
conda install -c anaconda scipy==1.2.1
conda install -c anaconda scikit-learn==0.21.3

# visualization
conda install -c conda-forge matplotlib==3.1.1
conda install -c anaconda seaborn==0.9.0

# dynet
conda install -c anaconda cython
conda install -c intel mkl
pip install dynet

conda install -c conda-forge jupyterlab
ipython kernel install --user --name=d3m_esrnn
conda install -c anaconda pylint
conda install -c anaconda pyyaml
conda deactivate
