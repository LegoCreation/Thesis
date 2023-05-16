# Bachelor's Thesis 2023

## Installation

```bash
git clone https://github.com/LegoCreation/Thesis.git
cd Thesis/
conda create --name new-ml
conda activate new-ml
pip3 install -r requirements.txt
```
## Usage guide
```
mkdir data
cd data
mkdir output_data_new
cp [src]/C6H6.xyz .
mkdir splitted_data
cd splitted data
csplit -f 'C6H6.xyz_' -b '%d' ../C6H6.xyz '/^[[:space:]]**12$/' '{*}'
python3 generate_cmr.py
```

This will now create suitable dataset for further Machine Learning experiments.

KRR.py and GPR.py contain all the necessary classes. KRR.ipynb and GPR.ipynb contain the functions to plot the results.
