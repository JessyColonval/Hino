![Python](https://img.shields.io/badge/python-3.10+-blue)
![License](https://img.shields.io/badge/license-Apache%202.0-green)
![Tests](https://github.com/JessyColonval/Hino/actions/workflows/python-app.yml/badge.svg)

# Hunting Inside N-Quantiles of Outliers (Hino)

Source code of the Hino algorithm presented during 'The 29th Pacific-Asia Conference on Knowledge Discovery and Data Mining' conference in the Special Session 'DSFA: Learning on Complex Data'.


## ⚙️ How the Hino algorithm works

Hino's algorithm is based on similar principles to those used by the Interquartile Range (IQR) outlier detection approach.
Hino splits the points of a dataset into several quantiles and observes which are too often isolated to consider them as outliers.

Thus, a point can be isolated in two different ways:
- in space, i.e. it is distant from all other points;
- according to its pairs, i.e. that it is distant from other points with the same class.

Concretely, for each quantile, Hino observes the direct adjacent quantiles to ensure:
- they are not empty;
- at least one of these two quantiles contains at least one point of the same class as the ones in the current quantile.

Otherwise, points in the current quantile that do not meet the two conditions have their isolation scores incremented by 1.
And when an isolation score of a point is too high, then this point is considered an as outlier.


## 🗂️ Project tree structure

```bash
Hino/
├── .github/
│   └── workflows/            # YAML files for GitHub CI/CD Actions
│       └── python-app.yml    # Runs unit tests after each commit
├── src/                      # Main project source code
│   ├── __init__.py           # Makes the folder importable as a module
│   ├── hino.py               # Hino algorithm
│   └── quantile.py           # General quantile management
├── test/                     # Unit tests
│   ├── __init__.py
│   ├── test_hino.py
│   └── test_quantile.py
├── .gitignore                # Files/folders ignored by Git
├── LICENCE                   # The license applied to the project
├── README.md                 # Project description
├── example.py                # Python script illustrating the use of the project
└── requirements.txt          # Project dependencies
```


## 📄 Licence

This project is licensed under [Apache 2.0](https://www.apache.org/licenses/LICENSE-2.0).
You are free to use, modify and distribute this software, provided you comply with the terms of the license. 
See the [LICENSE](./LICENSE) file for the full license text.
