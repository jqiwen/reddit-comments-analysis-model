import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score
from sklearn.ensemble import RandomForestClassifier
import os

def load_data():
    train_data = pd.read_csv("./training_data/train_data.csv")
    test_data = pd.read_csv("./training_data/test_data.csv")
    val_data = pd.read_csv("./training_data/val_data.csv")
    
    return train_data, test_data, val_data

class logistic_regression_model():
    def __init__(self):
        self.train_data, self.test_data, self.val_data = load_data()
        
    def TFIDF_Comment(self):
        vectorizer = TfidfVectorizer(stop_words='english', max_features=5000, ngram_range=(1,2))
        self.X_train = vectorizer.fit_transform(self.train_data['Comment'].fillna(''))
        self.X_test = vectorizer.transform(self.test_data['Comment'].fillna(''))
        self.X_val = vectorizer.transform(self.val_data['Comment'].fillna(''))
                
    def train(self):
        self.model = LogisticRegression(max_iter=1000, solver='lbfgs', class_weight='balanced')
        self.model.fit(self.X_train, self.train_data['Ground_Truth_Label'])

    def predict(self):
        self.y_pred_test = self.model.predict(self.X_test)
        self.y_pred_val = self.model.predict(self.X_val)

    def show_performance(self): 
        test_acc = accuracy_score(self.test_data['Ground_Truth_Label'], self.y_pred_test)
        val_acc = accuracy_score(self.val_data['Ground_Truth_Label'], self.y_pred_val)
        report = classification_report(self.test_data['Ground_Truth_Label'], self.y_pred_test)

        # save to txt file
        os.makedirs('./outputs', exist_ok=True)
        with open('./outputs/baseline_results.txt', 'a', encoding='utf-8') as f:
            f.write("\n\n---------------------Logistic Regression Results---------------------\n")
            f.write(f"Test Accuracy: {test_acc:.4f}\n")
            f.write(f"Validation Accuracy: {val_acc:.4f}\n\n")
            f.write("Classification Report on Test Data:\n")
            f.write(report)
        
class Random_Forest_Model():
    def __init__(self):
        self.train_data, self.test_data, self.val_data = load_data()
        
    def TFIDF_Comment(self):
        vectorizer = TfidfVectorizer(stop_words='english', max_features=5000, ngram_range=(1,2))
        self.X_train = vectorizer.fit_transform(self.train_data['Comment'].fillna(''))
        self.X_test = vectorizer.transform(self.test_data['Comment'].fillna(''))
        self.X_val = vectorizer.transform(self.val_data['Comment'].fillna(''))
                
    def train(self):
        self.model = RandomForestClassifier(class_weight='balanced')
        self.model.fit(self.X_train, self.train_data['Ground_Truth_Label'])

    def predict(self):
        self.y_pred_test = self.model.predict(self.X_test)
        self.y_pred_val = self.model.predict(self.X_val)

    def show_performance(self): 
        test_acc = accuracy_score(self.test_data['Ground_Truth_Label'], self.y_pred_test)
        val_acc = accuracy_score(self.val_data['Ground_Truth_Label'], self.y_pred_val)
        report = classification_report(self.test_data['Ground_Truth_Label'], self.y_pred_test)

        # save to txt file
        os.makedirs('./outputs', exist_ok=True)
        with open('./outputs/baseline_results.txt', 'a', encoding='utf-8') as f:
            f.write("\n\n---------------------Random Forest Results---------------------\n")
            f.write(f"Test Accuracy: {test_acc:.4f}\n")
            f.write(f"Validation Accuracy: {val_acc:.4f}\n\n")
            f.write("Classification Report on Test Data:\n")
            f.write(report)

def main():
    print("Training Logistic Regression ... ")
    lr_model = logistic_regression_model()
    lr_model.TFIDF_Comment()
    lr_model.train()
    lr_model.predict()
    lr_model.show_performance()
    
    print("Training Random Forest ... ")
    rf_model = Random_Forest_Model()
    rf_model.TFIDF_Comment()
    rf_model.train()
    rf_model.predict()
    rf_model.show_performance()
    