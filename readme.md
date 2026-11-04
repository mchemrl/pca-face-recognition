# Face Recognition with PCA

## overview

**question:** how well can PCA (eigenfaces) + a simple classifier identify which of 40 people is in a face photo, and how many PCA components are enough?

**data:** the olivetti faces dataset: 400 grayscale photos (64x64 pixels), 40 people, 10 photos each. It is loaded through `sklearn.datasets.fetch_olivetti_faces`.

**approach:** a small empirical study that follows eleven steps, from the question to a reproducible report. We compare simple baselines on raw pixels against PCA-based pipelines, using stratified 5-fold cross-validation (every person appears in both train and test), and report accuracy and macro F1 as mean and std across folds. 

**repo layout:**
```
repo/
  main.ipynb        # the main notebook: question, data, validation, metrics, baseline, preprocessing, models, comparison, error analysis, conclusions
  src/
    common.py       # seed, cv splitter, data loader, evaluate()
  requirements.txt  # pinned dependencies
  readme.md
```

## requirements and installation

Using python 3.10 or newer, create an environment and install the dependencies:

```bash
python -m venv .venv

# linux / macos
source .venv/bin/activate
# windows
.venv\Scripts\activate

pip install -r requirements.txt
```

## running the notebook

1. open a terminal in the repo root
2. start jupyter:
   ```bash
   jupyter notebook main.ipynb
   ```
3. run all cells from top to bottom (kernel > restart and run all).

reproducibility: the random seed (`seed = 42`) and the cross-validation split are defined once in `src/common.py`, so every model is scored on identical folds and the results are the same on every run.

## results

fill in later

**comparison table** (5-fold cross-validation, mean ± std):

| model | accuracy | macro f1 |
|---|---|---|
| dummy (most frequent) | ___ | ___ |
| knn raw pixels | ___ | ___ |
| logreg raw pixels | ___ | ___ |
| pca50 + knn | ___ | ___ |
| ... | ___ | ___ |

**key findings:**
- ___
- ___

**limitations:**
- ___

## authors

- ___
- ___
