# scripts/generate_figures.py
"""
Generate ROC and confusion matrix PNGs from reports CSV/JSON.
Requires matplotlib and pandas.
"""
import pandas as pd, matplotlib.pyplot as plt
from pathlib import Path
REPORTS=Path('reports')
roc_csv = REPORTS / 'roc_arcface.csv'
if roc_csv.exists():
    df = pd.read_csv(roc_csv)
    plt.figure(figsize=(6,6))
    plt.plot(df['fpr'], df['tpr'], label='ArcFace')
    plt.xlabel('False Positive Rate'); plt.ylabel('True Positive Rate'); plt.title('ROC'); plt.legend()
    plt.grid(True)
    plt.savefig(REPORTS/'roc_arcface.png', dpi=300)
    print('Saved ROC PNG')
else:
    print('No roc_arcface.csv found to plot.')
