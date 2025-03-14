
from training_model.data_analysis import main as annotation_analysis
from training_model.trained_model import main as trained_model
from training_model.baseline_model import main as baseline_model

# 2 Annotation Analysis: Cohen's Kappa & Adjudication
annotation_analysis()

# 3.1 Simple Baseline: Random Baseline
baseline_model()

# 3.1 Trained Model: Logistic Regression & Random Forest
trained_model()
