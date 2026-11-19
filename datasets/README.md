# Dataset provenance

Both datasets are distributed by the UCI Machine Learning Repository under the [Creative Commons Attribution 4.0 International license](https://creativecommons.org/licenses/by/4.0/). The gzip files retain the source text rows and a final newline.

| Bundled file | Original source | Rows | Uncompressed SHA-256 |
|---|---|---:|---|
| `bike_hour.csv.gz` | Hadi Fanaee-T and João Gama. [Bike Sharing](https://doi.org/10.24432/C5W894). [Hourly CSV mirror](https://raw.githubusercontent.com/muditp19/UCI_Bike-sharing-dataset/master/hour.csv). | 17,379 | `b03a2d02e8c10f435c43c7f0b358b7e34a003afea53dbc37f0183f2763295133` |
| `letter_recognition.data.gz` | David Slate. [Letter Recognition](https://doi.org/10.24432/C5ZP40). [Data file](https://archive.ics.uci.edu/ml/machine-learning-databases/letter-recognition/letter-recognition.data). | 20,000 | `2b89f3602cf768d3c8355267d2f13f2417809e101fc2b5ceee10db19a60de6e2` |

Bike Sharing uses cyclical hour, weekday, and month features, working-day and holiday flags, year, weather indicators, and recorded temperature, apparent temperature, humidity, and wind speed. The observation order is chronological. The train, validation and test counts are 11,122, 2,781, and 3,476. The predictors exclude casual and registered rental counts. The task estimates hourly demand from its recorded covariates.

Letter Recognition uses the 16 numeric image descriptors to predict the 26 uppercase letter labels. The study groups identical feature vectors before partitioning.
