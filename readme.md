# Bachelor's Thesis 2023

## Installation

```bash
pip3 install -r requirements.txt
mkdir data
cd data
mkdir output_data_new
cp [src]/C6H6.xyz .
mkdir splitted_data
cd splitted data
csplit -f 'C6H6.xyz_' -b '%d' ../C6H6.xyz '/^[[:space:]]**12$/' '{*}'
```
