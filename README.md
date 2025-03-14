### ✅ To Run the Project:
Open and run `model.py` from the project home directory.  
All results will be automatically saved under the `outputs/` folder.

### 📁 Outputs Directory Overview

There are **3 main result files** saved under the `outputs/` folder:

- `data_annotation_with_ground_truth.xlsx` — Final annotated dataset with adjudicated ground truth labels.
- `ground_truth_label_distribution_pie.png` — Pie chart showing the distribution of ground truth labels.
- `baseline_results.txt` — Evaluation results from both baseline and trained models.

### 📁 training_model Directory Overview

There are **3 Python files** under the `training_model/` directory:

#### 🔸 `data_analysis.py` — Annotation Analysis
Includes:
- **Cohen's Kappa**  
  → Result will be printed in the terminal.
- **Ground Truth Adjudication**  
  → Saved as `outputs/data_annotation_with_ground_truth.xlsx`.
- **Ground Truth Label Distribution (Pie Chart)**  
  → Saved as `outputs/ground_truth_label_distribution_pie.png`.

#### 🔸 `baseline_model.py` — Simple Baseline Model
- Implements a **Random Baseline classifier**.
- Evaluation results are saved in:  
  → `outputs/baseline_results.txt`

#### 🔸 `trained_model.py` — Trained Models
- Includes two trained models:
  - **Logistic Regression**
  - **Random Forest**
- Evaluation results are saved in:  
  → `outputs/baseline_results.txt`

---

If needed, you can add a link at the end:
> 📎 *Codabench Task Link:* [Igh]  
> 📎 *Presentation Slides:* [hjh]
