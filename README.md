# IPL Innings-Break Winner Prediction

This is a simple machine learning project to predict the winner of an IPL match.
The target is whether the team batting first wins the match or not.

The prediction is made at innings break, so the model can use first innings
score, wickets, run rate, team records, toss details, venue and city.

I made this project with simple models so the complete process is easy to
understand:

- Linear Regression baseline
- Logistic Regression
- Random Forest
- Extra Trees
- Gradient Boosting
- XGBoost, if the library is available

## Project Flow

1. Load IPL ball-by-ball dataset.
2. Convert ball-by-ball data into match-level data.
3. Clean team names and keep current IPL teams.
4. Create innings-break features like first innings score and run rate.
5. Create historical features like team win rate, recent form and head-to-head record.
6. Train simple ML models.
7. Compare the models using accuracy, F1 score and ROC-AUC.
8. Save the best model.

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
- EDA charts
- model comparison table with accuracy and F1 score
- classification report for the best model

The best trained model is saved in:

```text
models/best_ipl_model.pkl
```

## Note

This is not a pure pre-match prediction system. It predicts the result at
innings break after the first innings score is known. I chose this setup because
pure pre-match IPL prediction is very noisy, while innings-break prediction gives
a stronger and more useful machine learning problem.
=======
# ipl-match-winner-prediction
