from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy
from scipy import stats
import seaborn as sns

sns.set_theme(style='whitegrid', context='notebook')

DATA_PATH = Path("data/academic_survival_longitudinal.csv")
df = pd.read_csv(DATA_PATH)

numeric_correlations = df.select_dtypes(include='number').corr()

fig, ax = plt.subplots(figsize=(12,16))
sns.heatmap(numeric_correlations, annot=True, fmt='0.2f', cmap='vlag', center=0, square=True, ax=ax)
ax.set_title('Correlation Matrix')
plt.tight_layout()
plt.show()

