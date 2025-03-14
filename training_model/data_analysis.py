import os
import glob
import pandas as pd
from collections import defaultdict
from sklearn.metrics import cohen_kappa_score
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

def main():
    # import all annotated data, and merge into one file

    print("Merging annotated data ... ")

    folder_path = './annotation_data'
    files = glob.glob(os.path.join(folder_path, 'dataset_*.xlsx'))

    label_mapping = {
        'high': 'High-Value',
        'High-Value': 'High-Value',
        'mid': 'Mid-Value',
        'Mid-Value': 'Mid-Value',
        'low': 'Low-Value',
        'Low-Value': 'Low-Value',
        'noisy': 'Noisy',
        'Noisy': 'Noisy'
    }

    records = []
    for file in files:
        df = pd.read_excel(file)
        df = df.rename(columns={"Label (High-Value / Mid-Value / Low-Value / Noisy)": "Label"})
        df['Label'] = df['Label'].map(label_mapping)

        for _, row in df.iterrows():
            records.append((row['Post Title'], row['Comment'], row['Label']))

    merged_dict = defaultdict(list)
    for title, comment, label in records:
        key = (title, comment)
        merged_dict[key].append(label)

    merged_rows = []
    for (title, comment), labels in merged_dict.items():
        row = {
            'Post Title': title,
            'Comment': comment
        }
        for i, lbl in enumerate(labels):
            row[f'Label{i+1}'] = lbl
        merged_rows.append(row)

    merged_df = pd.DataFrame(merged_rows)

    # Drop rows where both Label1 and Label2 are missing
    if 'Label1' in merged_df.columns and 'Label2' in merged_df.columns:
        merged_df = merged_df.dropna(subset=['Label1', 'Label2'], how='all')

    # export single data file
    os.makedirs('./outputs', exist_ok=True)
    os.makedirs('./outputs/annotation_alalysis', exist_ok=True)
    merged_output_path = 'outputs/annotation_alalysis/merged_annotation_data.xlsx'
    merged_df.to_excel(merged_output_path, index=False)

    # ------------------------------------------------------------------------------------------------------------

    # Using Cohen's Kappa to assess the consistency of annotations

    print("Calculation Cohen's Kappa ... ")

    label1 = 'Label1'
    label2 = 'Label2'

    df_pair = merged_df.dropna(subset=[label1, label2]).copy()
    # pair_output_path = 'outputs/annotation_alalysis/cohen_pair_data.xlsx'
    # df_pair.to_excel(pair_output_path, index=False)

    if len(df_pair) == 0:
        print(f"There is only 1 label, can't calculate Cohen's Kappa")
    else:
        le = LabelEncoder()
        combined_labels = pd.concat([df_pair[label1], df_pair[label2]])
        le.fit(combined_labels)
        l1 = le.transform(df_pair[label1])
        l2 = le.transform(df_pair[label2])

        # Calculate Kappa
        score = cohen_kappa_score(l1, l2)
        print(f"Score of Cohen’s Kappa: {score:.3f}")

        # Add label_adjudication column
        df_pair['Label_adjudication'] = df_pair[label1].where(df_pair[label1] == df_pair[label2])

        # Export updated data
        adjudication_output_path = 'outputs/annotation_alalysis/cohen_pair_data_with_adjudication.xlsx'
        df_pair.to_excel(adjudication_output_path, index=False)

    # ------------------------------------------------------------------------------------------------------------

    # ground trugh
    print("Adding Ground Truth Label ...")

    # Generate Ground Truth Column in merged_df
    adjudication_df = pd.read_excel('annotation_data/data_adjudication.xlsx')

    def get_ground_truth(row):
        if pd.isna(row['Label2']):
            return row['Label1']
        elif pd.isna(row['Label1']):
            return row['Label2']
        else:
            matched = adjudication_df[
                (adjudication_df['Post Title'] == row['Post Title']) &
                (adjudication_df['Comment'] == row['Comment'])
            ]
            if not matched.empty:
                return matched.iloc[0]['Label_adjudication']
            else:
                return None

    merged_df['Ground_Truth_Label'] = merged_df.apply(get_ground_truth, axis=1)
    merged_df = merged_df[['Post Title', 'Comment', 'Ground_Truth_Label']].dropna()

    # Export merged data with ground truth
    ground_truth_output_path = 'outputs/data_annotation_with_ground_truth.xlsx'
    merged_df.to_excel(ground_truth_output_path, index=False)

    # ------------------------------------------------------------------------------------------------------------

    # Visualize distribution of Ground Truth Labels
    print("Visualizing the distribution of Ground truth label ...")

    label_counts = merged_df['Ground_Truth_Label'].value_counts()

    # Pie Chart
    plt.figure(figsize=(6,6))
    label_counts.plot(kind='pie', autopct='%1.1f%%', startangle=90)
    plt.title('Ground Truth Label Distribution')
    plt.ylabel('')
    plt.tight_layout()
    plt.savefig('outputs/ground_truth_label_distribution_pie.png')
    plt.close()

    # ------------------------------------------------------------------------------------------------------------

    data = merged_df.reset_index(drop=True)
    data['ID'] = data.index + 1
    data = data[['ID', 'Post Title', 'Comment', 'Ground_Truth_Label']]

    # split data into train/val/test（70/10/20）
    train_val_data, test_data = train_test_split(data, test_size=0.2, random_state=42, stratify=data['Ground_Truth_Label'])
    train_data, val_data = train_test_split(train_val_data, test_size=0.125, random_state=42, stratify=train_val_data['Ground_Truth_Label'])

    # export csv file
    os.makedirs('training_data', exist_ok=True)
    train_data.to_csv('training_data/train_data.csv', index=False)
    val_data.to_csv('training_data/val_data.csv', index=False)
    test_data.to_csv('training_data/test_data.csv', index=False)

