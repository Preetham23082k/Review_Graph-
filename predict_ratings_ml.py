import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import accuracy_score, mean_absolute_error, mean_squared_error, cohen_kappa_score
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt
import numpy as np
from sklearn.utils import resample

# Import the data
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler

data = pd.read_csv("machine_learning_features_node2vec_5_extra_sentiment_features.csv")
data = data[['node2vec_0', 'node2vec_1', 'node2vec_2', 'node2vec_3', 'node2vec_4',
             'min_sentiment', 'max_sentiment', 'rating','avg_sentiment']]

# Find the minimum class size
min_size = data['rating'].value_counts().min()

# Convert the date column into more useful features
# data['year'] = data['date'].dt.year
# data['month'] = data['date'].dt.month
# data['day'] = data['date'].dt.day
# data['day_of_week'] = data['date'].dt.dayofweek  # Monday=0, Sunday=6
# data['weekofyear'] = data['date'].dt.isocalendar().week
# data['is_weekend'] = data['day_of_week'].apply(lambda x: 1 if x >= 5 else 0)

import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, mean_absolute_error, mean_squared_error, cohen_kappa_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.dummy import DummyClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.utils import resample

# Set sampling method: "oversampling" or "none"
sampling_method = "none"  # change to "none" for no sampling

# Load your dataset
data = pd.read_csv("machine_learning_features_node2vec5.csv")
data = data[['node2vec_0', 'node2vec_1', 'node2vec_2', 'node2vec_3', 'node2vec_4',
             'rating', 'avg_sentiment']]

# Define models to test
models = {
    "Random Forest": RandomForestClassifier(),
    "Logistic Regression": LogisticRegression(max_iter=1000, solver='lbfgs'),
    "Neural Network (MLP)": MLPClassifier(hidden_layer_sizes=(100,), max_iter=500, random_state=42),
    "Dummy (Most Frequent)": DummyClassifier(strategy="most_frequent")
}

y = data['rating']
X = data.drop(['rating'], axis=1)

results = {model_name: {"accuracy": [], "mae": [], "rmse": [], "kappa": []} for model_name in models}

kf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)

for fold, (train_idx, test_idx) in enumerate(kf.split(X, y)):
    print(f"\n--- Fold {fold+1} ---")
    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
    y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

    if sampling_method == "oversampling":
        X_train_resampled = pd.DataFrame()
        y_train_resampled = pd.Series(dtype=int)
        max_size = y_train.value_counts().max()
        for rating in y_train.unique():
            X_class = X_train[y_train == rating]
            y_class = y_train[y_train == rating]
            X_class_resampled, y_class_resampled = resample(X_class, y_class, replace=True, n_samples=max_size, random_state=42)
            X_train_resampled = pd.concat([X_train_resampled, X_class_resampled])
            y_train_resampled = pd.concat([y_train_resampled, y_class_resampled])
        X_train_resampled = X_train_resampled.sample(frac=1, random_state=42)
        y_train_resampled = y_train_resampled.sample(frac=1, random_state=42)
    else:
        X_train_resampled = X_train
        y_train_resampled = y_train

    for model_name, model in models.items():
        X_train_fold = X_train_resampled.copy()
        X_test_fold = X_test.copy()

        # Scale for models that require it
        if model_name in ["Logistic Regression", "Neural Network (MLP)"]:
            scaler = StandardScaler()
            X_train_fold = scaler.fit_transform(X_train_fold)
            X_test_fold = scaler.transform(X_test_fold)

        model.fit(X_train_fold, y_train_resampled)
        y_pred = model.predict(X_test_fold)

        results[model_name]["accuracy"].append(accuracy_score(y_test, y_pred))
        results[model_name]["mae"].append(mean_absolute_error(y_test, y_pred))
        results[model_name]["rmse"].append(mean_squared_error(y_test, y_pred))
        results[model_name]["kappa"].append(cohen_kappa_score(y_test, y_pred))

# Average results
avg_results_dict = {model: {metric: np.mean(vals) for metric, vals in metrics.items()}
                    for model, metrics in results.items()}
avg_results = pd.DataFrame.from_dict(avg_results_dict, orient='index')

# Print averaged results
print(f"\n=== Averaged 10-Fold CV Results ({sampling_method.title()}) ===")
print(avg_results.round(4))


# Plotting all metrics per model
avg_results.plot(kind='bar', figsize=(12, 6))
plt.title("Model Comparison: Accuracy, MAE, RMSE, Cohen's Kappa")
plt.ylabel("Score")
plt.xticks(rotation=45)
plt.grid(True, linestyle='--', axis='y')
plt.tight_layout()
plt.show()

# Ensure y_test and y_pred are NumPy arrays or Series
y_test = pd.Series(y_test)
y_pred = pd.Series(y_pred)

# Count actual and predicted ratings
actual_counts = y_test.value_counts().sort_index()
predicted_counts = y_pred.value_counts().sort_index()

# Align indices (ensure all ratings 1–5 are present)
all_ratings = range(1, 6)
actual_counts = actual_counts.reindex(all_ratings, fill_value=0)
predicted_counts = predicted_counts.reindex(all_ratings, fill_value=0)

# Bar width and positions
x = np.arange(len(all_ratings))
bar_width = 0.35

# Plot
plt.figure(figsize=(8, 6))
plt.bar(x - bar_width/2, actual_counts, width=bar_width, label='Actual', color='skyblue')
plt.bar(x + bar_width/2, predicted_counts, width=bar_width, label='Predicted', color='salmon')
plt.xticks(x, all_ratings)
plt.xlabel('Rating')
plt.ylabel('Count')
plt.title('Actual vs Predicted Review Ratings')
plt.legend()
plt.grid(True, axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()
