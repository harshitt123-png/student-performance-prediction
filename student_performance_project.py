"""
Student Performance Prediction - Python / Machine Learning part
Run:  python student_performance_project.py
If you have your own CSV, set USE_OWN_DATA = True and change the path/target below.
"""
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt, seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

USE_OWN_DATA = False
OWN_PATH, TARGET = "my_data.csv", "final_score"

# ---------- 1. Data ----------
if USE_OWN_DATA:
    df = pd.read_csv(OWN_PATH)
else:
    # Sample (synthetic) dataset, 400 students. Replace with real data if you have it.
    rng = np.random.default_rng(42); n = 400
    df = pd.DataFrame({
        "study_hours":   rng.normal(14, 5, n).clip(1, 30).round(1),
        "attendance":    rng.normal(80, 10, n).clip(40, 100).round(1),
        "previous_score":rng.normal(62, 14, n).clip(20, 100).round(1),
        "sleep_hours":   rng.normal(6.8, 1.1, n).clip(4, 10).round(1),
        "absences":      rng.poisson(5, n),
        "internet_access": rng.choice(["Yes", "No"], n, p=[.8, .2]),
        "parental_education": rng.choice(["School", "Graduate", "Postgraduate"], n, p=[.4, .4, .2]),
    })
    edu = df.parental_education.map({"School": 0, "Graduate": 2, "Postgraduate": 4})
    df["final_score"] = (8 + 1.1*df.study_hours + 0.25*df.attendance + 0.45*df.previous_score
                         + 1.0*df.sleep_hours - 0.6*df.absences + edu
                         + (df.internet_access == "Yes")*2 + rng.normal(0, 4, n)).clip(0, 100).round(1)
    TARGET = "final_score"
df.to_csv("student_data.csv", index=False)

# ---------- 2. Cleaning ----------
print("Shape:", df.shape); print(df.isna().sum()); print(df.describe().round(2))
df = df.drop_duplicates().dropna()

# ---------- 3. EDA ----------
sns.set_style("whitegrid")
plt.figure(figsize=(6,4)); sns.histplot(df[TARGET], kde=True, color="#2E5C8A")
plt.title("Distribution of Final Score"); plt.tight_layout(); plt.savefig("fig1_distribution.png", dpi=150); plt.close()
plt.figure(figsize=(7,5)); sns.heatmap(df.select_dtypes("number").corr(), annot=True, fmt=".2f", cmap="coolwarm")
plt.title("Correlation Matrix"); plt.tight_layout(); plt.savefig("fig2_correlation.png", dpi=150); plt.close()
plt.figure(figsize=(6,4)); sns.scatterplot(data=df, x="study_hours", y=TARGET, hue="internet_access")
plt.title("Study Hours vs Final Score"); plt.tight_layout(); plt.savefig("fig3_scatter.png", dpi=150); plt.close()

# ---------- 4. Modelling ----------
X = pd.get_dummies(df.drop(columns=TARGET), drop_first=True); y = df[TARGET]
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42)
models = {"Linear Regression": LinearRegression(),
          "Random Forest": RandomForestRegressor(n_estimators=200, random_state=42)}
rows = []
for name, m in models.items():
    m.fit(X_tr, y_tr); p = m.predict(X_te)
    cv = cross_val_score(m, X, y, cv=5, scoring="r2").mean()
    rows.append([name, r2_score(y_te, p), mean_absolute_error(y_te, p),
                 np.sqrt(mean_squared_error(y_te, p)), cv])
res = pd.DataFrame(rows, columns=["Model", "R2", "MAE", "RMSE", "CV_R2"]).round(3)
print(res); res.to_csv("model_results.csv", index=False)

# ---------- 5. Plots of results ----------
rf = models["Random Forest"]; lr = models["Linear Regression"]
imp = pd.Series(rf.feature_importances_, index=X.columns).sort_values()
plt.figure(figsize=(6,4)); imp.plot.barh(color="#2E5C8A"); plt.title("Random Forest Feature Importance")
plt.tight_layout(); plt.savefig("fig4_importance.png", dpi=150); plt.close()
plt.figure(figsize=(5,5)); plt.scatter(y_te, lr.predict(X_te), alpha=.6)
plt.plot([y.min(), y.max()], [y.min(), y.max()], "r--"); plt.xlabel("Actual"); plt.ylabel("Predicted")
plt.title("Linear Regression: Actual vs Predicted"); plt.tight_layout(); plt.savefig("fig5_actual_vs_pred.png", dpi=150); plt.close()
pd.Series(lr.coef_, index=X.columns).round(3).to_csv("linear_coefficients.csv")
imp.sort_values(ascending=False).round(3).to_csv("feature_importance.csv")
print("Done. Files saved.")
