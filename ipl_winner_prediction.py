"""
IPL Innings-Break Winner Prediction

This project predicts whether the team batting first will win the match.
The prediction is made at innings break, so first innings score information is
allowed. This gives a more useful and realistic model than only pre-match data.
"""

from pathlib import Path
import pickle
import warnings

import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import ExtraTreesClassifier, GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, f1_score, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

warnings.filterwarnings("ignore")


DATA_DIR = Path("data")
MODEL_DIR = Path("models")
DATA_FILE = DATA_DIR / "IPL.csv"


def load_dataset():
    """Load IPL data from local folder, KaggleHub cache, or KaggleHub download."""
    if DATA_FILE.exists():
        print("Loading dataset from data/IPL.csv")
        return pd.read_csv(DATA_FILE, low_memory=False)

    cache_files = sorted(
        Path.home().glob(".cache/kagglehub/datasets/chaitu20/ipl-dataset2008-2025/**/IPL.csv")
    )
    if cache_files:
        print("Loading dataset from KaggleHub cache")
        return pd.read_csv(cache_files[-1], low_memory=False)

    print("Dataset not found locally, downloading from KaggleHub...")
    import kagglehub

    path = Path(kagglehub.dataset_download("chaitu20/ipl-dataset2008-2025"))
    return pd.read_csv(path / "IPL.csv", low_memory=False)


def clean_season(value):
    """Convert season formats like 2007/08 into a single year."""
    value = str(value)
    season_map = {"2007/08": 2008, "2009/10": 2010, "2020/21": 2021}
    return season_map.get(value, int(value[:4]))


def prepare_match_data(df):
    """Convert ball-by-ball rows into one clean row per match."""
    match_df = df.groupby("match_id").first().reset_index()

    match_cols = [
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
    match_df = match_df[match_cols].copy()

    # Short names make the notebook tables easier to read.
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
    match_df = match_df[
        match_df["batting_team"].isin(current_teams)
        & match_df["bowling_team"].isin(current_teams)
        & match_df["match_won_by"].isin(current_teams)
    ].copy()

    match_df["season"] = match_df["season"].apply(clean_season)
    match_df["date"] = pd.to_datetime(match_df["date"])
    match_df = match_df.sort_values("date").reset_index(drop=True)

    match_df["team1"] = match_df["batting_team"]
    match_df["team2"] = match_df["bowling_team"]
    match_df["team1_won"] = (match_df["match_won_by"] == match_df["team1"]).astype(int)

    match_df["toss_is_team1"] = (match_df["toss_winner"] == match_df["team1"]).astype(int)
    match_df["toss_decision_bat"] = (match_df["toss_decision"] == "bat").astype(int)

    knockout_stages = {"Final", "Qualifier 1", "Qualifier 2", "Eliminator", "Semi Final", "Elimination Final"}
    match_df["is_knockout"] = match_df["stage"].isin(knockout_stages).astype(int)

    return match_df


def add_first_innings_features(df, match_df):
    """Add innings-break features like first innings score and run rate."""
    first_innings = (
        df[df["innings"] == 1]
        .groupby("match_id")
        .agg(
            first_innings_runs=("runs_total", "sum"),
            first_innings_balls=("valid_ball", "sum"),
            first_innings_wickets=("player_out", lambda x: x.notna().sum()),
            first_innings_boundaries=("runs_batter", lambda x: ((x == 4) | (x == 6)).sum()),
        )
        .reset_index()
    )

    match_df = match_df.merge(first_innings, on="match_id", how="left")
    match_df["first_innings_run_rate"] = match_df["first_innings_runs"] / (
        match_df["first_innings_balls"] / 6
    )

    return match_df


def add_history_features(match_df):
    """Create past-record features without using future matches."""
    team_stats = {}
    h2h_stats = {}
    venue_stats = {}
    venue_scores = {}
    recent_results = {}

    history_cols = {
        "team1_win_rate": [],
        "team2_win_rate": [],
        "team1_recent_form": [],
        "team2_recent_form": [],
        "h2h_team1_rate": [],
        "venue_team1_rate": [],
        "venue_team2_rate": [],
        "venue_avg_first_score": [],
    }

    def team_win_rate(team):
        stats = team_stats.get(team, {"matches": 0, "wins": 0})
        return 0.5 if stats["matches"] == 0 else stats["wins"] / stats["matches"]

    def recent_form(team):
        results = recent_results.get(team, [])
        return 0.5 if not results else sum(results[-5:]) / len(results[-5:])

    def venue_win_rate(team, venue):
        stats = venue_stats.get((team, venue), {"matches": 0, "wins": 0})
        return 0.5 if stats["matches"] == 0 else stats["wins"] / stats["matches"]

    def h2h_rate(team1, team2):
        key = tuple(sorted([team1, team2]))
        stats = h2h_stats.get(key, {"matches": 0, "wins": {}})
        return 0.5 if stats["matches"] == 0 else stats["wins"].get(team1, 0) / stats["matches"]

    for _, row in match_df.iterrows():
        team1 = row["team1"]
        team2 = row["team2"]
        venue = row["venue"]
        winner = team1 if row["team1_won"] == 1 else team2

        # These values are calculated before updating with the current match.
        # This is important because otherwise the model will see the answer.
        history_cols["team1_win_rate"].append(team_win_rate(team1))
        history_cols["team2_win_rate"].append(team_win_rate(team2))
        history_cols["team1_recent_form"].append(recent_form(team1))
        history_cols["team2_recent_form"].append(recent_form(team2))
        history_cols["h2h_team1_rate"].append(h2h_rate(team1, team2))
        history_cols["venue_team1_rate"].append(venue_win_rate(team1, venue))
        history_cols["venue_team2_rate"].append(venue_win_rate(team2, venue))
        history_cols["venue_avg_first_score"].append(np.mean(venue_scores.get(venue, [160])))

        for team in [team1, team2]:
            team_stats.setdefault(team, {"matches": 0, "wins": 0})
            team_stats[team]["matches"] += 1
            team_stats[team]["wins"] += int(team == winner)

            venue_stats.setdefault((team, venue), {"matches": 0, "wins": 0})
            venue_stats[(team, venue)]["matches"] += 1
            venue_stats[(team, venue)]["wins"] += int(team == winner)

            recent_results.setdefault(team, []).append(int(team == winner))

        key = tuple(sorted([team1, team2]))
        h2h_stats.setdefault(key, {"matches": 0, "wins": {}})
        h2h_stats[key]["matches"] += 1
        h2h_stats[key]["wins"][winner] = h2h_stats[key]["wins"].get(winner, 0) + 1

        venue_scores.setdefault(venue, []).append(row["first_innings_runs"])

    for col, values in history_cols.items():
        match_df[col] = values

    match_df["win_rate_diff"] = match_df["team1_win_rate"] - match_df["team2_win_rate"]
    match_df["recent_form_diff"] = match_df["team1_recent_form"] - match_df["team2_recent_form"]
    match_df["h2h_diff"] = match_df["h2h_team1_rate"] - 0.5
    match_df["venue_rate_diff"] = match_df["venue_team1_rate"] - match_df["venue_team2_rate"]
    match_df["score_vs_venue_avg"] = (
        match_df["first_innings_runs"] - match_df["venue_avg_first_score"]
    )

    return match_df


def build_dataset(df):
    """Run full preprocessing and return final match dataset."""
    match_df = prepare_match_data(df)
    match_df = add_first_innings_features(df, match_df)
    match_df = add_history_features(match_df)

    needed = ["first_innings_runs", "first_innings_run_rate", "city"]
    match_df = match_df.dropna(subset=needed).reset_index(drop=True)

    return match_df


def get_feature_lists():
    numeric_features = [
        "season",
        "toss_is_team1",
        "toss_decision_bat",
        "is_knockout",
        "first_innings_runs",
        "first_innings_wickets",
        "first_innings_run_rate",
        "first_innings_boundaries",
        "venue_avg_first_score",
        "score_vs_venue_avg",
        "win_rate_diff",
        "recent_form_diff",
        "h2h_diff",
        "venue_rate_diff",
    ]

    categorical_features = ["team1", "team2", "venue", "city"]
    return numeric_features, categorical_features


def get_models(preprocessor):
    """Create simple and explainable models for comparison."""
    models = {
        "Linear Regression Baseline": Pipeline(
            [("preprocessor", preprocessor), ("model", LinearRegression())]
        ),
        "Logistic Regression": Pipeline(
            [
                ("preprocessor", preprocessor),
                ("model", LogisticRegression(max_iter=2000, class_weight="balanced", random_state=42)),
            ]
        ),
        "Random Forest": Pipeline(
            [
                ("preprocessor", preprocessor),
                (
                    "model",
                    RandomForestClassifier(
                        n_estimators=400,
                        max_depth=12,
                        min_samples_leaf=3,
                        class_weight="balanced",
                        random_state=42,
                    ),
                ),
            ]
        ),
        "Extra Trees": Pipeline(
            [
                ("preprocessor", preprocessor),
                (
                    "model",
                    ExtraTreesClassifier(
                        n_estimators=400,
                        max_depth=12,
                        min_samples_leaf=3,
                        class_weight="balanced",
                        random_state=42,
                    ),
                ),
            ]
        ),
        "Gradient Boosting": Pipeline(
            [
                ("preprocessor", preprocessor),
                (
                    "model",
                    GradientBoostingClassifier(
                        n_estimators=200,
                        max_depth=3,
                        learning_rate=0.05,
                        random_state=42,
                    ),
                ),
            ]
        ),
    }

    try:
        from xgboost import XGBClassifier

        models["XGBoost"] = Pipeline(
            [
                ("preprocessor", preprocessor),
                (
                    "model",
                    XGBClassifier(
                        n_estimators=150,
                        max_depth=3,
                        learning_rate=0.06,
                        subsample=0.9,
                        colsample_bytree=0.9,
                        eval_metric="logloss",
                        random_state=42,
                    ),
                ),
            ]
        )
    except Exception as error:
        print(f"XGBoost is not available here, so I am skipping it. Reason: {error}")

    return models


def predict_for_metrics(model, X_test, model_name):
    """Handle both classifier models and the linear regression baseline."""
    if model_name == "Linear Regression Baseline":
        raw_score = model.predict(X_test)
        probability = np.clip(raw_score, 0, 1)
        prediction = (probability >= 0.5).astype(int)
        return prediction, probability

    prediction = model.predict(X_test)
    probability = model.predict_proba(X_test)[:, 1]
    return prediction, probability


def train_and_evaluate(match_df):
    numeric_features, categorical_features = get_feature_lists()
    features = numeric_features + categorical_features

    train_df = match_df[match_df["season"] <= 2022].copy()
    test_df = match_df[match_df["season"] >= 2023].copy()

    X_train = train_df[features]
    y_train = train_df["team1_won"]
    X_test = test_df[features]
    y_test = test_df["team1_won"]

    preprocessor = ColumnTransformer(
        [
            ("num", StandardScaler(), numeric_features),
            ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features),
        ]
    )

    models = get_models(preprocessor)
    results = []
    fitted_models = {}

    for name, model in models.items():
        print(f"Training {name}...")
        model.fit(X_train, y_train)
        y_pred, y_prob = predict_for_metrics(model, X_test, name)

        results.append(
            {
                "Model": name,
                "Accuracy": round(accuracy_score(y_test, y_pred), 3),
                "F1 Score": round(f1_score(y_test, y_pred), 3),
                "ROC AUC": round(roc_auc_score(y_test, y_prob), 3),
            }
        )
        fitted_models[name] = model

    results_df = pd.DataFrame(results).sort_values("Accuracy", ascending=False)
    best_name = results_df.iloc[0]["Model"]

    return results_df, fitted_models[best_name], best_name, X_test, y_test


def main():
    df = load_dataset()
    print("Raw dataset shape:", df.shape)

    match_df = build_dataset(df)
    print("Final match dataset shape:", match_df.shape)
    print("Seasons:", match_df["season"].min(), "to", match_df["season"].max())

    results_df, best_model, best_name, X_test, y_test = train_and_evaluate(match_df)
    print("\nModel comparison:")
    print(results_df.to_string(index=False))

    best_pred, _ = predict_for_metrics(best_model, X_test, best_name)
    print("\nBest model:", best_name)
    print(classification_report(y_test, best_pred, target_names=["Team2 won", "Team1 won"]))

    MODEL_DIR.mkdir(exist_ok=True)
    with open(MODEL_DIR / "best_ipl_model.pkl", "wb") as file:
        pickle.dump(best_model, file)

    print("Best model saved at:", MODEL_DIR / "best_ipl_model.pkl")


if __name__ == "__main__":
    main()
