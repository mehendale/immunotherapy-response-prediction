import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# 1. LOAD DATA

import zipfile
import pandas as pd

zip_path = r"C:\Users\USER\OneDrive\Desktop\expression.zip"
extract_path = r"C:\Users\USER\OneDrive\Desktop\expression_data"

with zipfile.ZipFile(zip_path, "r") as zip_ref:
    zip_ref.extractall(extract_path)


expr = pd.read_csv(extract_path + r"\expression.csv", index_col=0)

#meta data loading 

meta_path = r"C:\Users\USER\OneDrive\Desktop\metadata.txt"

meta = pd.read_csv(meta_path, sep="\t")


# 2. CLEAN + FORMAT

expr = expr.T

expr.index = expr.index.str.strip()
meta["sample"] = meta["sample"].str.strip()

# Merge expression with metadata
df = expr.merge(meta, left_index=True, right_on="sample")


# 3. LABEL CREATION
# Convert response into 0/1
# Change column name depending on your metadata

df["response"] = df["response"].map({
    "Responder": 1,
    "Non-responder": 0
})

# Drop unnecessary columns
df = df.dropna(subset=["response"])

# 4. PREPROCESSING

X = df.drop(["sample", "response"], axis=1)
y = df["response"]

# Keep numeric only
X = X.select_dtypes(include=[np.number])


# 5. TRAIN TEST SPLIT

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 6. MODEL TRAINING


model = RandomForestClassifier(n_estimators=100)
model.fit(X_train, y_train)

# 7. EVALUATION

y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

print(f"\nModel Accuracy: {accuracy:.2f}")


# 8. FEATURE IMPORTANCE

importances = model.feature_importances_
genes = X.columns

feat_df = pd.DataFrame({
    "gene": genes,
    "importance": importances
}).sort_values(by="importance", ascending=False)

top10 = feat_df.head(10)
# 9. TOP GENES PLOT


plt.figure()
plt.barh(top10["gene"], top10["importance"])
plt.title("Top 10 Important Genes")
plt.gca().invert_yaxis()
plt.show()

# 10. HEATMAP

top_genes = top10["gene"].values

heatmap_data = X[top_genes].iloc[:30]

plt.figure()
sns.heatmap(heatmap_data, cmap="coolwarm")
plt.title("Heatmap of Top Genes")
plt.show()

# 11. CORRELATION MATRIX

plt.figure()
sns.heatmap(X[top_genes].corr(), annot=True)
plt.title("Gene Correlation Matrix")
plt.show()

# 12. SAMPLE PREDICTION + EXPLANATION

sample = X_test.iloc[0]

prediction = model.predict([sample])[0]
prob = model.predict_proba([sample])[0][1]

gene_info = {
    "CD74": "Antigen presentation",
    "BIRC3": "Apoptosis regulation",
    "PDCD1": "Immune checkpoint",
    "CTLA4": "T-cell inhibition"
}

print("\n🔬 Prediction Report")
print("----------------------")

if prediction == 1:
    print(f"Prediction: Responder ({prob:.2f})")
else:
    print(f"Prediction: Non-Responder ({prob:.2f})")

high_genes = []
low_genes = []

print("\nTop Influencing Genes:")

for gene in top10["gene"][:5]:
    value = sample[gene]
    mean_val = X[gene].mean()

    if value > mean_val:
        high_genes.append(gene)
        direction = "↑"
    else:
        low_genes.append(gene)
        direction = "↓"

    meaning = gene_info.get(gene, "Biological role")

    print(f"- {gene} {direction} ({meaning})")

# 13. FINAL EXPLANATION

print("\n🧠 Explanation:")

print(f"High expression: {', '.join(high_genes)}")
print(f"Low expression: {', '.join(low_genes)}")
print("High immune activation genes suggest stronger immune response.")
print("Checkpoint and apoptosis variation may explain response differences.")
