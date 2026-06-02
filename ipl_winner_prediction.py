"""
IPL Match Winner Prediction

This project predicts whether the team batting first will win an IPL match.
I kept the models simple because the main aim is to understand the complete
machine learning workflow properly.
"""

from pathlib import Path
import pickle
import warnings

import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

warnings.filterwarnings("ignore")


DATA_DIR = Path("data")
MODEL_DIR = Path("models")
DATA_FILE = DATA_DIR / "IPL.csv"


def load_dataset():
    """Load the IPL dataset from local folder or download it from Kaggle."""
    if DATA_FILE.exists():
        print("Loading dataset from local data folder...")
        return pd.read_csv(DATA_FILE, low_memory=False)

    print("Local dataset not found, downloading from Kaggle...")
    import kagglehub

    dataset_path = Path(kagglehub.dataset_download("chaitu20/ipl-dataset2008-2025"))
    csv_path = dataset_path / "IPL.csv"
    return pd.read_csv(csv_path, low_memory=False)


def clean_season(value):
    """Convert season values like 2007/08 or 2020/21 into normal years."""
    value = str(value)

    season_map = {
        "2007/08": 2008,
        "2009/10": 2010,
        "2020/21": 2021,
    }

    if value in season_map:
        return season_map[value]

    # For normal values like 2019 or 2024, only the first 4 digits are needed.
    return int(value[:4])


def prepare_match_data(df):
    """Convert ball-by-ball data into one row per match."""
    print("Preparing match level data...")

    match_df = df.groupby("match_id").first().reset_index()

    needed_cols = [
        "match_id",
        "date",
        "season",
        "venue",
        "city",
        "batting_team",
        "bowling_team",
        "toss_winner",
        "toss_decision",
        "match_won_by",
        "stage",
    ]

    match_df = match_df[needed_cols].copy()

    # I am using short team names because they are easier to read in the output.
    team_map = {
        "Chennai Super Kings": "CSK",
        "Delhi Capitals": "DC",
        "Delhi Daredevils": "DC",
        "Gujarat Titans": "GT",
        "Kolkata Knight Riders": "KKR",
        "Mumbai Indians": "MI",
        "Punjab Kings": "PBKS",
        "Kings XI Punjab": "PBKS",
        "Rajasthan Royals": "RR",
        "Royal Challengers Bangalore": "RCB",
        "Royal Challengers Bengaluru": "RCB",
        "Sunrisers Hyderabad": "SRH",
        "Lucknow Super Giants": "LSG",
    }

    for col in ["batting_team", "bowling_team", "toss_winner", "match_won_by"]:
        match_df[col] = match_df[col].map(team_map)

    current_teams = {"CSK", "DC", "GT", "KKR", "LSG", "MI", "PBKS", "RCB", "RR", "SRH"}

    # Old teams are removed to keep the project focused on current IPL teams.
    match_df = match_df[
        match_df["batting_team"].isin(current_teams)
        & match_df["bowling_team"].isin(current_teams)
        & match_df["match_won_by"].isin(current_teams)
    ].copy()

    match_df["season"] = match_df["season"].apply(clean_season)
    match_df["date"] = pd.to_datetime(match_df["date"])
    match_df = match_df.sort_values("date").reset_index(drop=True)

    # In this dataset, batting_team is the team batting first and bowling_team is chasing.
    match_df["team1"] = match_df["batting_team"]
    match_df["team2"] = match_df["bowling_team"]

    # Target column: 1 means team batting first won, 0 means chasing team won.
    match_df["team1_won"] = (match_df["match_won_by"] == match_df["team1"]).astype(int)

    match_df["toss_is_team1"] = (match_df["toss_winner"] == match_df["team1"]).astype(int)
    match_df["toss_decision_bat"] = (match_df["toss_decision"] == "bat").astype(int)

    knockout_stages = {
        "Final",
        "Qualifier 1",
        "Qualifier 2",
        "Eliminator",
        "Semi Final",
        "Elimination Final",
    }
    match_df["is_knockout"] = match_df["stage"].isin(knockout_stages).astype(int)

    return match_df


def add_history_features(match_df):
    """Add simple past-performance features without using future match data."""
    print("Creating history based features...")

    team_stats = {}
    h2h_stats = {}
    venue_stats = {}

    team1_rates = []
    team2_rates = []
    h2h_rates = []
    venue_team1_rates = []
    venue_team2_rates = []

    def get_team_rate(team):
        stats = team_stats.get(team, {"matches": 0, "wins": 0})
        if stats["matches"] == 0:
            return 0.5
        return stats["wins"] / stats["matches"]

    def get_h2h_rate(team1, team2):
        key = tuple(sorted([team1, team2]))
        stats = h2h_stats.get(key)
        if not stats:
            return 0.5
        total = stats["matches"]
        if total == 0:
            return 0.5
        return stats["wins"].get(team1, 0) / total

    def get_venue_rate(team, venue):
        key = (team, venue)
        stats = venue_stats.get(key, {"matches": 0, "wins": 0})
        if stats["matches"] == 0:
            return 0.5
        return stats["wins"] / stats["matches"]

    for _, row in match_df.iterrows():
        team1 = row["team1"]
        team2 = row["team2"]
        venue = row["venue"]
        team1_won = row["team1_won"]
        winner = team1 if team1_won == 1 else team2

        # These features are calculated before updating with the current match.
        # This prevents data leakage because the model cannot see the result in advance.
        team1_rates.append(get_team_rate(team1))
        team2_rates.append(get_team_rate(team2))
        h2h_rates.append(get_h2h_rate(team1, team2))
        venue_team1_rates.append(get_venue_rate(team1, venue))
        venue_team2_rates.append(get_venue_rate(team2, venue))

        for team in [team1, team2]:
            team_stats.setdefault(team, {"matches": 0, "wins": 0})
            team_stats[team]["matches"] += 1
            if team == winner:
                team_stats[team]["wins"] += 1

            venue_stats.setdefault((team, venue), {"matches": 0, "wins": 0})
            venue_stats[(team, venue)]["matches"] += 1
            if team == winner:
                venue_stats[(team, venue)]["wins"] += 1

        key = tuple(sorted([team1, team2]))
        h2h_stats.setdefault(key, {"matches": 0, "wins": {}})
        h2h_stats[key]["matches"] += 1
        h2h_stats[key]["wins"][winner] = h2h_stats[key]["wins"].get(winner, 0) + 1

    match_df["team1_win_rate"] = team1_rates
    match_df["team2_win_rate"] = team2_rates
    match_df["h2h_team1_rate"] = h2h_rates
    match_df["venue_team1_rate"] = venue_team1_rates
    match_df["venue_team2_rate"] = venue_team2_rates

    # Difference features usually work better than using both team values separately.
    match_df["win_rate_diff"] = match_df["team1_win_rate"] - match_df["team2_win_rate"]
    match_df["venue_rate_diff"] = match_df["venue_team1_rate"] - match_df["venue_team2_rate"]
    match_df["h2h_diff"] = match_df["h2h_team1_rate"] - 0.5

    return match_df


def train_test_split_by_time(match_df):
    """Use older seasons for training and recent seasons for testing."""
    train = match_df[match_df["season"] <= 2022].copy()
    test = match_df[match_df["season"] >= 2023].copy()

    # Small fallback in case someone uses a smaller dataset.
    if len(train) == 0 or len(test) == 0:
        split_index = int(len(match_df) * 0.8)
        train = match_df.iloc[:split_index].copy()
        test = match_df.iloc[split_index:].copy()

    return train, test


def get_models(preprocessor):
    """Create all models used in this project."""
    models = {
        "Linear Regression Baseline": Pipeline(
            steps=[
                ("preprocessor", preprocessor),
                ("model", LinearRegression()),
            ]
        ),
        "Logistic Regression": Pipeline(
            steps=[
                ("preprocessor", preprocessor),
                ("model", LogisticRegression(max_iter=1000, random_state=42)),
            ]
        ),
        "Random Forest": Pipeline(
            steps=[
                ("preprocessor", preprocessor),
                (
                    "model",
                    RandomForestClassifier(
                        n_estimators=150,
                        max_depth=6,
                        min_samples_leaf=5,
                        random_state=42,
                    ),
                ),
            ]
        ),
    }

    try:
        from xgboost import XGBClassifier

        models["XGBoost"] = Pipeline(
            steps=[
                ("preprocessor", preprocessor),
                (
                    "model",
                    XGBClassifier(
                        n_estimators=120,
                        max_depth=3,
                        learning_rate=0.08,
                        subsample=0.9,
                        colsample_bytree=0.9,
                        eval_metric="logloss",
                        random_state=42,
                    ),
                ),
            ]
        )
    except Exception as error:
        print(f"XGBoost is not available on this system, so I am skipping it. Reason: {error}")

    return models


def predict_for_metrics(model, X_test, model_name):
    """Return class prediction and probability score for a model."""
    if model_name == "Linear Regression Baseline":
        raw_score = model.predict(X_test)
        probability = np.clip(raw_score, 0, 1)
        prediction = (probability >= 0.5).astype(int)
        return prediction, probability

    prediction = model.predict(X_test)

    if hasattr(model, "predict_proba"):
        probability = model.predict_proba(X_test)[:, 1]
    else:
        probability = prediction

    return prediction, probability


def main():
    df = load_dataset()
    print("Raw dataset shape:", df.shape)

    match_df = prepare_match_data(df)
    match_df = add_history_features(match_df)

    print("Final match dataset shape:", match_df.shape)
    print("Seasons:", match_df["season"].min(), "to", match_df["season"].max())

    numeric_features = [
        "season",
        "toss_is_team1",
        "toss_decision_bat",
        "is_knockout",
        "win_rate_diff",
        "venue_rate_diff",
        "h2h_diff",
    ]
    categorical_features = ["team1", "team2", "venue", "city"]

    features = numeric_features + categorical_features
    target = "team1_won"

    train, test = train_test_split_by_time(match_df)

    X_train = train[features]
    y_train = train[target]
    X_test = test[features]
    y_test = test[target]

    print("\nTrain matches:", len(train), "| Test matches:", len(test))
    print("Train seasons:", train["season"].min(), "to", train["season"].max())
    print("Test seasons:", test["season"].min(), "to", test["season"].max())

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), numeric_features),
            ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features),
        ]
    )

    models = get_models(preprocessor)
    results = []
    fitted_models = {}

    print("\nTraining models...\n")

    for name, model in models.items():
        model.fit(X_train, y_train)
        y_pred, y_prob = predict_for_metrics(model, X_test, name)

        accuracy = accuracy_score(y_test, y_pred)
        roc_auc = roc_auc_score(y_test, y_prob)

        results.append(
            {
                "Model": name,
                "Accuracy": round(accuracy, 3),
                "ROC_AUC": round(roc_auc, 3),
            }
        )
        fitted_models[name] = model

    results_df = pd.DataFrame(results).sort_values("Accuracy", ascending=False)
    print(results_df.to_string(index=False))

    best_model_name = results_df.iloc[0]["Model"]
    best_model = fitted_models[best_model_name]

    print("\nBest Model:", best_model_name)
    best_pred, _ = predict_for_metrics(best_model, X_test, best_model_name)
    print("\nClassification Report:")
    print(classification_report(y_test, best_pred, target_names=["Team2 won", "Team1 won"]))

    MODEL_DIR.mkdir(exist_ok=True)
    with open(MODEL_DIR / "best_ipl_model.pkl", "wb") as file:
        pickle.dump(best_model, file)

    print("\nBest model saved at:", MODEL_DIR / "best_ipl_model.pkl")


if __name__ == "__main__":
    main()
