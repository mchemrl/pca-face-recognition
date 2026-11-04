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

**comparison table** (5-fold stratified cross-validation, mean +- std across folds; the three PCA models use 50 components):

| model | accuracy | macro f1 |
|---|---|---|
| dummy (most frequent) | 0.025 +- 0.000 | 0.001 +- 0.000 |
| knn raw pixels | 0.933 +- 0.013 | 0.924 +- 0.015 |
| logreg raw pixels | 0.980 +- 0.006 | 0.979 +- 0.007 |
| pca50 + knn | 0.930 +- 0.013 | 0.924 +- 0.016 |
| pca50 + svm (rbf) | 0.932 +- 0.032 | 0.926 +- 0.037 |
| pca50 + logistic regression | 0.980 +- 0.006 | 0.979 +- 0.007 |

**accuracy vs number of PCA components** (mean +- std across folds):

| k | pca + knn | pca + svm | pca + logreg |
|---|---|---|---|
| 5 | 0.735 +- 0.058 | 0.600 +- 0.078 | 0.622 +- 0.063 |
| 10 | 0.885 +- 0.022 | 0.823 +- 0.046 | 0.875 +- 0.029 |
| 20 | 0.927 +- 0.027 | 0.912 +- 0.036 | 0.963 +- 0.009 |
| 50 | 0.930 +- 0.014 | 0.932 +- 0.036 | 0.980 +- 0.007 |
| 100 | 0.935 +- 0.016 | 0.943 +- 0.030 | 0.980 +- 0.007 |
| 300 | 0.932 +- 0.014 | 0.945 +- 0.026 | 0.982 +- 0.007 |

The full table, the curves and the error analysis are in `main.ipynb`.

**question:** how well can PCA (eigenfaces) + a simple classifier identify which of 40 people is in a face photo, and how many components are enough?

**key findings:**
1. **Identification works well.** PCA (50 components) + logistic regression reaches 0.980 +- 0.007 accuracy. PCA + kNN and PCA + SVM reach about 0.93-0.94.
2. **PCA loses almost nothing.** At 50 components the accuracy equals that of the same classifier on raw pixels (logistic regression 0.980 vs 0.980, kNN 0.930 vs 0.933), while using about 80 times fewer features than the 4096 pixels.
3. **How many components are enough.** Accuracy rises quickly up to about 20-30 components and is flat from about 50. 50 components keep 87.4% of the variance, so the variance is not needed in full for identification.
4. **Model ranking.** Logistic regression is better than kNN and SVM by about 4-5 points for k >= 20, which is larger than its std. kNN and SVM cannot be distinguished from each other.
5. **Errors.** The best model makes 8 errors out of 400 photos. In general, there is no group of people that the model systematically mixes up.

**limitations:**
- **Small data.** There are only 10 photos per person, so each fold trains on 8 photos per person and tests on 2. The uncertainty is wide.
- **Error analysis is visual.** With 8 errors, the patterns are just visual impressions and not tested causes.
- **Uniform conditions.** All photos are frontal, cropped and centered. In real conditions, the results would probably be worse.

## authors

- Polina Vladymyrska
- ___
