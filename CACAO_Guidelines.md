# Using COMPASS as a simulated bench for CACAO

- [CACAO installation](#cacao-installation)
- [Anaconda installation](#anaconda-installation)
- [Using pyImageStreamIO to read/write python data from/into CACAO SHM](#using-pyimagestreamio-to-readwrite-python-data-frominto-cacao-shm)
- [COMPASS installation](#compass-installation)
- [Summary environment variables](#summary-environment-variables)
- [Usage](#usage)
- [Configure CACAO to use COMPASS data](#configure-cacao-to-use-compass-data)

## CACAO installation

source: https://github.com/CACAO-org/CACAO

```bash
cd $HOME
git clone --recursive https://github.com/cacao-org/cacao cacao
cd cacao
autoreconf -vif
./configure
make
make install
export CACAO_ROOT=$HOME/cacao
```

## Anaconda installation

For an easier maintenance, I recommend the use of anaconda.
You can also use virtualenv to have a python sandbox, but I can't help you.

```bash
cd $HOME
export CONDA_ROOT=$HOME/miniconda3
wget https://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash Miniconda3-latest-Linux-x86_64.sh -b -p $CONDA_ROOT
export PATH=$CONDA_ROOT/bin:$PATH
```

## Using pyImageStreamIO to read/write python data from/into CACAO SHM

```bash
cd $HOME
git clone --recursive https://github.com/pybind/pybind11.git
export PYBIND11_ROOT=$HOME/pybind11
git clone -b dev https://github.com/milk-org/pyImageStreamIO.git
cd pyImageStreamIO
make
make install_conda
```

## COMPASS installation

```bash
conda install -c compass compass -y
git clone -b cacao https://github.com/ANR-COMPASS/shesha.git
export SHESHA_ROOT=$HOME/shesha
export PYTHONPATH=$SHESHA_ROOT/src:$PYTHONPATH
export PYTHONDONTWRITEBYTECODE=1
```

note: conda main channel is compiled with CUDA 9.1, for previous version please use:

```bash
conda install -c compass/label/cuda90 compass -y
conda install -c compass/label/cuda80 compass -y
```

## Summary environment variables

In your .bashrc, you should have defined:

```bash
export CACAO_ROOT=$HOME/cacao
export PYBIND11_ROOT=$HOME/pybind11
export CONDA_ROOT=$HOME/miniconda3
export PATH=$CONDA_ROOT/bin:$PATH
export SHESHA_ROOT=$HOME/shesha
export PYTHONPATH=$SHESHA_ROOT/src:$PYTHONPATH
export PYTHONDONTWRITEBYTECODE=1
```

## Usage

```bash
cd $SHESHA_ROOT
ipython -i widgets/widget_ao.py data/par/CACAO/scao_pyrhr_40x40.py -- --cacao -d 0
```

note: the flag ``-d 0`` impose compass to use GPU0 only.

More information on the widget: https://anr-compass.github.io/compass/manual.html#5-using-the-gui

If you don't need displays, you can also use the close_loop.py script

## Configure CACAO to use COMPASS data

COMPASS provides simulated frames in the SHM names ``compass_wfs0``, ``compass_wfs1``, ... In SCAO, we will use only ``compass_wfs0``.

Simulated DM shape is computed from command vectors written in the SHM ``compass_dm0``, ``compass_dm1``, ... In SCAO, we will use only ``compass_dm0``.

``compass_wfs0`` and ``compass_dm0`` are defined in the ``data/par/CACAO/scao_pyrhr_40x40.py``:

- ``compass_wfs0`` is a 128x128 float-frame (coded on 32bits)
- ``compass_dm0`` is a 45x45 float-frame (coded on 32bits)

In the parameter file, we use a non-modulated pyramid ( ``pyr_ampl=0`` &  ``pyr_npts=1`` ).

More info: https://anr-compass.github.io/compass/manual.html#param_wfs
