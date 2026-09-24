# ==========================================
# CodeAlpha - Titanic Exploratory Data Analysis
# Author: Kashish Bhadane
# ==========================================

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# ---------- Load Dataset ----------
url = "https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv"
df = pd.read_csv(url)

# ---------- Basic Dataset Information ----------
print("=" * 50)
print(" TITANIC DATASET - EXPLORATORY DATA ANALYSIS ")
print("=" * 50)

print("\nFirst 5 Rows:")
print(df.head())

print("\nDataset Information:")
df.info()

print("\nMissing Values:")
print(df.isnull().sum())

print("\nStatistical Summary:")
print(df.describe())

# ---------- Set Chart Style ----------
sns.set_style("whitegrid")

# ==========================================
# Chart 1: Survival Count
# ==========================================
plt.figure(figsize=(6,4))
sns.countplot(data=df, x="Survived", hue="Survived",
              palette=["red", "green"], legend=False)
plt.title("Passenger Survival Count")
plt.xlabel("Survived (0 = No, 1 = Yes)")
plt.ylabel("Passengers")
plt.tight_layout()
plt.savefig("1_survival_count.png")
plt.close()

# ==========================================
# Chart 2: Age Distribution
# ==========================================
plt.figure(figsize=(6,4))
sns.histplot(df["Age"], bins=20, kde=True, color="skyblue")
plt.title("Age Distribution")
plt.xlabel("Age")
plt.ylabel("Passengers")
plt.tight_layout()
plt.savefig("2_age_distribution.png")
plt.close()

# ==========================================
# Chart 3: Gender Distribution
# ==========================================
plt.figure(figsize=(6,4))
sns.countplot(data=df, x="Sex", hue="Sex",
              palette=["steelblue", "hotpink"], legend=False)
plt.title("Gender Distribution")
plt.xlabel("Gender")
plt.ylabel("Passengers")
plt.tight_layout()
plt.savefig("3_gender_distribution.png")
plt.close()

# ==========================================
# Chart 4: Survival by Gender
# ==========================================
plt.figure(figsize=(6,4))
sns.countplot(data=df, x="Sex", hue="Survived",
              palette=["red", "green"])
plt.title("Survival by Gender")
plt.xlabel("Gender")
plt.ylabel("Passengers")
plt.legend(title="Survived")
plt.tight_layout()
plt.savefig("4_survival_by_gender.png")
plt.close()

# ==========================================
# Chart 5: Passenger Class Distribution
# ==========================================
plt.figure(figsize=(5,5))
df["Pclass"].value_counts().sort_index().plot(
    kind="pie",
    autopct="%1.1f%%",
    colors=["gold", "silver", "brown"]
)
plt.title("Passenger Class Distribution")
plt.ylabel("")
plt.tight_layout()
plt.savefig("5_passenger_class_pie.png")
plt.close()

# ==========================================
# Chart 6: Fare Distribution
# ==========================================
plt.figure(figsize=(6,4))
sns.histplot(df["Fare"], bins=30, kde=True, color="purple")
plt.title("Fare Distribution")
plt.xlabel("Fare")
plt.ylabel("Passengers")
plt.tight_layout()
plt.savefig("6_fare_distribution.png")
plt.close()

# ==========================================
# Chart 7: Age by Passenger Class
# ==========================================
plt.figure(figsize=(6,4))
sns.boxplot(data=df, x="Pclass", y="Age",
            hue="Pclass", palette="Set2", legend=False)
plt.title("Age by Passenger Class")
plt.xlabel("Passenger Class")
plt.ylabel("Age")
plt.tight_layout()
plt.savefig("7_age_by_class.png")
plt.close()

# ==========================================
# Chart 8: Correlation Heatmap
# ==========================================
plt.figure(figsize=(8,6))
sns.heatmap(
    df.corr(numeric_only=True),
    annot=True,
    cmap="coolwarm",
    linewidths=0.5
)
plt.title("Correlation Heatmap")
plt.tight_layout()
plt.savefig("8_correlation_heatmap.png")
plt.close()

# ---------- Completion Message ----------
print("\nEDA Completed Successfully!")
print("\nGenerated Files:")
print("1. 1_survival_count.png")
print("2. 2_age_distribution.png")
print("3. 3_gender_distribution.png")
print("4. 4_survival_by_gender.png")
print("5. 5_passenger_class_pie.png")
print("6. 6_fare_distribution.png")
print("7. 7_age_by_class.png")
print("8. 8_correlation_heatmap.png")
