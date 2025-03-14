import pandas as pd
import numpy as np
from sklearn.metrics import classification_report
import os

def load_data():
    train_data = pd.read_csv("./training_data/train_data.csv")
    test_data = pd.read_csv("./training_data/test_data.csv")
    val_data = pd.read_csv("./training_data/val_data.csv")
    
    return train_data, test_data, val_data

def main():
    train_data, test_data, val_data = load_data()
    np.random.seed(42)
    random_preds = np.random.choice(train_data['Ground_Truth_Label'].unique(), size=len(test_data))

    print("Training Random Baseline ... ")
    report = classification_report(test_data['Ground_Truth_Label'], random_preds)

    os.makedirs('./outputs', exist_ok=True)
    with open('./outputs/baseline_results.txt', 'w', encoding='utf-8') as f:
        f.write("---------------------Random Baseline Report--------------------- \n")
        f.write(report)
    
# main()
