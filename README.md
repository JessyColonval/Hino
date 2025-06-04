# Hunting Inside N-Quantiles of Outliers (Hino)

Source code of the Hino algorithm presented during 'The 29th Pacific-Asia Conference on Knowledge Discovery and Data Mining' conference in the Special Session 'DSFA: Learning on Complex Data'.

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
