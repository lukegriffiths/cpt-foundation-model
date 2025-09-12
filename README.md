# CPT foundation model

The CPT foundation model project aims to develop machine learning models for interpreting and analyzing cone penetration test (CPT) data. By leveraging a comprehensive dataset of CPT measurements, the project seeks to automate geotechnical parameter estimation and support engineering decision-making processes. The repository includes code, configuration files, and documentation to facilitate reproducible research and model deployment.

## Data

### CPT database

The dataset contains 1339 cone penetration tests (CPT, CPTu, SCPT, SCPTu) executed within Austria and Germany by the company Premstaller Geotechnik ZT GmbH.

https://www.tugraz.at/en/institutes/ibg/research/computational-geotechnics-group/database/

Oberhollenzer S., Premstaller M., Marte R., Tschuchnigg F., Erharter G.H., Marcher T. (2020): CPT dataset Premstaller Geotechnik. Data in Brief.

# To do

Remove auto loading and chunking of data from datamodule.
Improve training logging and evaluate results.

## Notes

Required to ensure conda's version of bundled libraries are used, fixing /lib/x86_64-linux-gnu/libstdc++.so.6: version `CXXABI_1.3.15' not found

    export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:$LD_LIBRARY_PATH"