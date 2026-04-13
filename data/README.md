# Data

This repository does not redistribute the original clinical report datasets.

Please obtain the datasets from their official sources:

- PIFIR: https://physionet.org/content/pifir/1.0.0/
- CHIFIR: https://physionet.org/content/corpus-fungal-infections/1.0.2/
- MIMIC-CXR: https://physionet.org/content/mimic-cxr/2.1.0/

After obtaining access, place the CSV files under `data/raw/`.

Dataset-specific label mapping is implemented in `src/rads/data.py`.
