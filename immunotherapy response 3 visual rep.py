import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from imblearn.over_sampling import SMOTE

# 1. DATA LOADING  
expr_path = r"C:\Users\USER\OneDrive\Desktop\expressions.gz"
meta_path = r"C:\Users\USER\OneDrive\Desktop\metadata.txt"

# Gene Map
gene_map = {
    '84059': 'CXCL13', '815': 'SERPINB9', '55320': 'TIGIT', '7450': 'TXN',
    '5294': 'PI3', '9807': 'STK17B', '4000': 'LMLN', '4291': 'MLF1', 
    '112714': 'KRT86', '11107': 'ZIC2', '929': 'CD8A', '2323': 'GZMB'
}

expr = pd.read_csv(expr_path, sep=',', index_col=0)
expr.columns = expr.columns.str.replace('"', '').str.strip()

sample_titles, response_values = [], []
with open(meta_path, 'r', encoding='utf-8', errors='ignore') as f:
    for line in f:
        if line.startswith('!Sample_title'):
            sample_titles = [s.replace('"', '').strip() for s in line.strip().split('\t')[1:]]
        if '!Sample_characteristics_ch1' in line and 'response' in line.lower():
            if not response_values:
                response_values = [r.replace('"', '').strip() for r in line.strip().split('\t')[1:]]

meta_map = pd.DataFrame({'Response': response_values}, index=sample_titles)
df = expr.T.merge(meta_map, left_index=True, right_index=True)
df = df[df['Response'] != 'response: UNK'].copy()
df['target'] = df['Response'].apply(lambda x: 1 if 'PRCR' in str(x).upper() else 0)

X = df.drop(['Response', 'target'], axis=1).apply(pd.to_numeric, errors='coerce').fillna(0)
y = df['target'].astype(int)

# 2. MODELING WITH SMOTE 
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
rf = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')

y_real, y_pred_total = [], []
for train_index, test_index in skf.split(X, y):
    X_train, X_test = X.iloc[train_index], X.iloc[test_index]
    y_train, y_test = y.iloc[train_index], y.iloc[test_index]
    X_train_res, y_train_res = SMOTE(random_state=42).fit_resample(X_train, y_train)
    rf.fit(X_train_res, y_train_res)
    y_pred_total.extend(rf.predict(X_test))
    y_real.extend(y_test)

#3. CREATIVE VISUALIZATION (Multi-Panel)
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

# Panel 1: Confusion Matrix
cm = confusion_matrix(y_real, y_pred_total)
sns.heatmap(cm, annot=True, fmt='d', cmap='Purples', ax=ax1,
            xticklabels=['Non-Resp', 'Resp'], yticklabels=['Non-Resp', 'Resp'])
ax1.set_title("Model Reliability (Confusion Matrix)")
ax1.set_ylabel('Actual')
ax1.set_xlabel('Predicted')

# Panel 2: Feature Importance with Names
importances = rf.feature_importances_
indices = np.argsort(importances)[-12:]
top_names = [gene_map.get(str(X.columns[i]), str(X.columns[i])) for i in indices]

ax2.barh(range(len(indices)), importances[indices], color='plum', align='center')
ax2.set_yticks(range(len(indices)))
ax2.set_yticklabels(top_names)
ax2.set_title("Top 12 Bio-Markers for Response")
ax2.set_xlabel("Importance Score")

plt.tight_layout()

# Panel 3: Volcano Plot Style (Scatter of Mean Expression vs Importance)
plt.figure(figsize=(10, 6))
mean_expr = X.mean()
plt.scatter(mean_expr, importances, alpha=0.5, color='grey')
# Highlight top genes
for i in indices:
    plt.scatter(mean_expr.iloc[i], importances[i], color='red')
    plt.text(mean_expr.iloc[i], importances[i], top_names[indices.tolist().index(i)], fontsize=9)

plt.title("Importance vs Expression Level (Volcano Signature)")
plt.xlabel("Mean Expression Level")
plt.ylabel("Predictive Importance")
plt.show()
