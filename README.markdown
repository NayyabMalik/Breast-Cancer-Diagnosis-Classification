# Breast Cancer Diagnosis Classification

This repository contains a machine learning project to classify breast tumors as malignant (1) or benign (0) using the Breast Cancer Wisconsin (Diagnostic) Dataset. The project involves data preprocessing, model training, and evaluation with a confusion matrix heatmap and classification report. It is part of my coursework at the National University of Modern Languages, Islamabad, submitted on October 31, 2024, under the supervision of Mam Iqra Nasem.

## Project Overview

The goal is to predict breast tumor diagnosis (malignant or benign) using 30 features derived from digitized images of fine needle aspirates of breast masses. The project includes:

- **Data Preprocessing**: Loading the dataset, encoding the `diagnosis` column, and dropping the empty `Unnamed: 32` column.
- **Modeling**: Training a machine learning model (e.g., logistic regression, SVM, or neural network) to classify tumors.
- **Evaluation**: Generating a confusion matrix heatmap and classification report to assess model performance.
- **Visualization**: Saving the confusion matrix as a heatmap in `static/images/`.

This project builds on concepts from my deep learning labs, particularly classification tasks using scikit-learn and TensorFlow.

## Dataset

- **Source**: Breast Cancer Wisconsin (Diagnostic) Dataset from the UCI Machine Learning Repository or Kaggle.
- **Size**: 569 samples, 33 columns (30 features, `id`, `diagnosis`, and `Unnamed: 32`).
- **Features**:
  - 30 numerical features (e.g., `radius_mean`, `texture_mean`, `perimeter_worst`, `concavity_se`) describing cell nuclei characteristics.
  - Features include mean, standard error (`_se`), and worst (`_worst`) values for 10 attributes (e.g., radius, texture, perimeter).
- **Target**:
  - `diagnosis`: Binary label (`M` = malignant = 1, `B` = benign = 0).
- **Notes**:
  - `Unnamed: 32` is empty (0 non-null) and dropped during preprocessing.
  - `id` is not used for modeling.

## Repository Structure

```
breast-cancer-classification/
├── data/
│   ├── data.csv                     # Breast Cancer Wisconsin dataset
├── notebooks/
│   ├── Breast Cancer Wisconsin (Diagnostic) Dataset.ipynb  # Jupyter notebook for analysis                
├── LICENSE                           # MIT License
├── README.md                         # This file
```

## Methodology

1. **Data Loading**:

   - Load the dataset (`data.csv`) using pandas.
   - Display initial data with `df.head()` and check structure with `df.info()`.

2. **Preprocessing**:

   - Encode `diagnosis` column: `M` (malignant) → 1, `B` (benign) → 0.
   - Drop the `Unnamed: 32` column due to all null values.
   - Verify unique values in `diagnosis` (expected: \[1, 0\]).

3. **Modeling**:

   - Split data into training (80%) and testing (20%) sets using `train_test_split`.
   - Train a classification model (e.g., logistic regression, SVM, or neural network; not shown in the provided notebook but assumed).
   - Generate predictions (`y_pred`) for the test set (`y_test`).

4. **Evaluation and Visualization**:

   - Compute a confusion matrix using `sklearn.metrics.confusion_matrix`.
   - Visualize the confusion matrix as a heatmap using seaborn and matplotlib.
   - Generate a classification report with precision, recall, and F1-score.
   - Save the heatmap to `static/images/confusion_matrix.png`.

## Results

- **Data Shapes**:
  - Total samples: 569.
  - Features: 30 (after dropping `id` and `Unnamed: 32`).
  - Training: \~455 samples (80%).
  - Testing: \~114 samples (20%).
- **Visualization**:
  - Confusion matrix heatmap (`confusion_matrix.png`) shows true positives, true negatives, false positives, and false negatives.
  - Classification report provides metrics like precision, recall, and F1-score (specific values depend on the model used).
- **Performance**: Model performance details (e.g., accuracy, F1-score) are not shown in the notebook; further evaluation is recommended.

## Related Coursework

This project builds on my deep learning labs, particularly:

- **Lab 3: CNN Classification** (`deep-learning-labs/lab_manuals/CNN_Classification.pdf`): Binary classification concepts, applicable to this dataset.
- **Lab 4: CNN Patterns** (`deep-learning-labs/lab_manuals/CNN_Patterns.pdf`): Feature preprocessing and model evaluation techniques.
- **Computer Vision Labs** (`computer-vision-labs/notebooks/`): Data visualization and preprocessing methods.

See the `deep-learning-labs` and `computer-vision-labs` repositories for details.

## Setup Instructions

1. **Clone the Repository**:

   ```bash
   git clone https://github.com/your-username/breast-cancer-classification.git
   cd breast-cancer-classification
   ```

2. **Install Dependencies**: Install Python libraries listed in `requirements.txt`:

   ```bash
   pip install -r requirements.txt
   ```

   Key libraries: `pandas`, `numpy`, `seaborn`, `matplotlib`, `scikit-learn`.

3. **Download Dataset**:

   - Place `data.csv` in the `data/` folder. The dataset can be sourced from UCI Machine Learning Repository or Kaggle.

4. **Run the Notebook**: Launch Jupyter Notebook and execute the analysis:

   ```bash
   jupyter notebook notebooks/Breast\ Cancer\ Wisconsin\ (Diagnostic)\ Dataset.ipynb
   ```

5. **View Visualizations**:

   - The confusion matrix heatmap is saved in `static/images/confusion_matrix.png`.

## Usage

1. **Load and Preprocess Data**:

   ```python
   import pandas as pd
   df = pd.read_csv("data/data.csv")
   df['diagnosis'] = df['diagnosis'].replace({'M': 1, 'B': 0})
   df = df.drop('Unnamed: 32', axis=1)
   ```

2. **Train and Evaluate Model** (Example with Logistic Regression):

   ```python
   from sklearn.model_selection import train_test_split
   from sklearn.linear_model import LogisticRegression
   from sklearn.metrics import confusion_matrix, classification_report
   import seaborn as sns
   import matplotlib.pyplot as plt
   
   X = df.drop(['id', 'diagnosis'], axis=1)
   y = df['diagnosis']
   X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
   
   model = LogisticRegression(max_iter=1000)
   model.fit(X_train, y_train)
   y_pred = model.predict(X_test)
   
   matrix = confusion_matrix(y_test, y_pred)
   sns.heatmap(matrix, annot=True, fmt="d")
   plt.title('Confusion Matrix')
   plt.xlabel('Predicted')
   plt.ylabel('True')
   plt.savefig("static/images/confusion_matrix.png")
   plt.show()
   print(classification_report(y_test, y_pred))
   ```

3. **Customize Model**:

   - Replace logistic regression with other models (e.g., SVM, Random Forest, or a neural network with TensorFlow) based on your coursework or preference.
   - Add feature scaling (e.g., `StandardScaler`) for better performance.

## Future Improvements

- **Model Implementation**: Include the full model training code (e.g., neural network architecture) in the notebook for clarity.
- **Feature Selection**: Apply techniques like PCA or correlation analysis to reduce dimensionality (30 features may include redundant ones).
- **Evaluation Metrics**: Report accuracy, ROC-AUC, or cross-validation scores in the notebook.
- **Visualizations**: Add more plots (e.g., feature importance, ROC curve) to `static/images/`.
- **Web Interface**: Develop a Flask-based interface (similar to `sales-forecasting/app.py`) for interactive predictions.
- **Hyperparameter Tuning**: Use GridSearchCV or RandomizedSearchCV to optimize model performance.

## Notes

- **File Size**: Use Git LFS for large files like `data.csv` or images (e.g., `git lfs track "*.csv" "*.png"`).
- **Dataset Availability**: Ensure `data.csv` is included or provide a link to the source in the README.
- **Privacy**: If submitting for coursework, verify with your instructor (e.g., Mam Iqra Nasem) if public sharing is allowed.

## License

This repository is licensed under the MIT License. See the LICENSE file for details.