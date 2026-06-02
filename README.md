<<<<<<< HEAD
# IPL Match Winner Prediction

This is a simple machine learning project to predict the winner of an IPL match.
The target is whether the team batting first wins the match or not.

I made this project with simple models so the complete process is easy to
understand:

- Linear Regression baseline
- Logistic Regression
- Random Forest
- XGBoost, if the library is installed

## Project Flow

1. Load IPL ball-by-ball dataset.
2. Convert ball-by-ball data into match-level data.
3. Clean team names and keep current IPL teams.
4. Create basic historical features like team win rate and head-to-head record.
5. Train simple ML models.
6. Compare the models using accuracy and ROC-AUC.
7. Save the best model.

## Dataset

The notebook/script downloads the dataset from Kaggle using `kagglehub`.

Dataset used:

`chaitu20/ipl-dataset2008-2025`

If you already have the dataset, you can also place `IPL.csv` inside a `data`
folder:

```text
ipl-match-winner-prediction/
  data/
    IPL.csv
```

## How To Run

Install the required libraries:

```bash
pip install -r requirements.txt
```

Run the Python file:

```bash
python ipl_winner_prediction.py
```

Or open the notebook:

```text
ipl_winner_prediction.ipynb
```

## Output

After running the project, it prints:

- dataset shape
- model comparison table
- classification report for the best model
- feature importance for tree based models

The best trained model is saved in:

```text
models/best_ipl_model.pkl
```

## Note

This is not meant to be a perfect cricket prediction system. IPL matches depend
on many things like playing XI, pitch, injuries, weather, and player form. This
project is mainly for learning how to build an end-to-end machine learning
workflow.
=======
# ipl-match-winner-prediction
>>>>>>> 7e948b3061aee810b8eb81d1cac304d56baf8f8f
