<div align="center">

# VCMR: Video-Conditioned Music Representations
  
</div>

This repository is the official implementation of the project.

## Installation

Clone the repository and install the dependencies. We recommend using a conda environment with Python 3.9+.
```
git clone https://github.com/klean2050/VCMR
cd VCMR 
pip install -r requirements.txt
```

You will also need to install the ``libsndfile`` library:
```
conda install -c conda-forge libsndfile
```

## Project Structure

```
VCMR/
├─ config/              # configuration files for each train session
├─ data/                # where datasets are downloaded (not included)
├─ tests/               # sample scripts to test functionalities
├─ vcmr/                # main project directory
│  ├─ index.js
├─ .gitignore
├─ export.py            # script to export model to ONNX
├─ LICENSE
├─ main.py              # driver script to run
├─ README.md
├─ requirements.txt
```
