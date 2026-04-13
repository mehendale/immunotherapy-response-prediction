import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

# 1. DATA LOADING & ROBUST CLEANING 
expr_path = r"C:\Users\USER\OneDrive\Desktop\expressions.gz"
meta_path = r"C:\Users\USER\OneDrive\Desktop\metadata.txt"

# Gene Mapping
gene_map = {
    '84059': 'CXCL13', '815': 'SERPINB9', '55320': 'TIGIT', '7450': 'TXN',
    '5294': 'PI3', '9807': 'STK17B', '4000': 'LMLN', '4291': 'MLF1', 
    '112714': 'KRT86', '11107': 'ZIC2', '929': 'CD8A', '2323': 'GZMB'
}

print("Loading data...")
expr = pd.read_csv(expr_path, sep=',', index_col=0)
# Clean Expression Headers (Remove quotes and whitespace)
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

# MERGE with cleaning
df = expr.T.merge(meta_map, left_index=True, right_index=True)

if df.empty:
    print("❌ Critical Match Failure! Let's check why:")
    print(f"Expr ID example: '{expr.columns[0]}'")
    print(f"Meta ID example: '{meta_map.index[0]}'")
    exit()

# Setup Columns
df['Timepoint'] = df.index.to_series().apply(lambda x: 'Pre' if 'PRE' in str(x).upper() else 'On')
df = df[df['Response'] != 'response: UNK'].copy()
df['target'] = df['Response'].apply(lambda x: 1 if 'PRCR' in str(x).upper() else 0)

# 2. ANALYTICS FUNCTION 
def run_full_analysis(data, title):
    print(f"\n{'='*20} {title} {'='*20}")
    
    X = data.drop(['Response', 'target', 'Timepoint'], axis=1).apply(pd.to_numeric, errors='coerce').fillna(0)
    y = data['target'].astype(int)
    
    if len(y.unique()) < 2:
        print(f"Skipping {title}: One class only (Responders: {sum(y)}, Non: {len(y)-sum(y)})")
        return None, None

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf.fit(X_train, y_train)
    
    print(classification_report(y_test, rf.predict(X_test), target_names=['Non-Responder', 'Responder']))
    print(f"ACCURACY: {accuracy_score(y_test, rf.predict(X_test)):.2f}")
    
    return rf, X

# 3. EXECUTION & VISUALIZATION 
results = run_full_analysis(df, "ALL SAMPLES")
if results[0] is not None:
    main_model, X_main = results
    
    # Analyze subgroups but don't crash if they fail
    run_full_analysis(df[df['Timepoint'] == 'Pre'], "PRE-TREATMENT")
    run_full_analysis(df[df['Timepoint'] == 'On'], "ON-TREATMENT")

    # A. TOP 10 GENES GRAPH
    importances = main_model.feature_importances_
    indices = np.argsort(importances)[-10:]
    top_ids = [X_main.columns[i] for i in indices]
    top_names = [gene_map.get(str(gid), str(gid)) for gid in top_ids]

    plt.figure(figsize=(10, 5))
    plt.barh(top_names, importances[indices], color='teal')
    plt.title("Top 10 Predictive Genes")
    plt.show()

    # B. HEATMAP (Z-Score normalized)
    plt.figure(figsize=(12, 6))
    heatmap_data = X_main[top_ids].copy()
    heatmap_data.columns = top_names
    heatmap_data = (heatmap_data - heatmap_data.mean()) / heatmap_data.std()
    
    sorted_df = heatmap_data.assign(target=df['target']).sort_values('target')
    sns.heatmap(sorted_df.drop('target', axis=1).T, cmap='RdBu_r', center=0)
    plt.title("Heatmap: Non-Responders (Left) vs Responders (Right)")
    plt.show()
else:
    print("Could not generate models due to data split issues.")