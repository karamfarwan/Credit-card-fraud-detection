# Credit Card Fraud Detection

This project aims to detect fraudulent transactions in credit card usage using machine learning techniques. The notebook implements various preprocessing, feature engineering, and classification models to predict whether a transaction is fraudulent.

## Project Structure

- **Data Preprocessing:** Handling missing values, scaling features, and balancing the dataset using techniques such as oversampling.
- **Feature Engineering:** Extracting relevant features for better prediction accuracy.
- **Modeling:** Several classification models are trained and evaluated, including Logistic Regression, Random Forest, and others.
- **Evaluation:** Performance metrics such as accuracy, precision, recall, F1-score, and confusion matrix are used to evaluate the models.

## Requirements

To run this notebook, you will need the following Python packages:

- `pandas`
- `numpy`
- `scikit-learn`
- `matplotlib`
- `seaborn`
- `nltk`
- `spacy` (Ensure that the `en_core_web_lg` model is installed by running `python -m spacy download en_core_web_lg`)
  
You can install these dependencies using:

```bash
pip install -r requirements.txt
```

Create a `requirements.txt` file:

```txt
pandas
numpy
scikit-learn
matplotlib
seaborn
nltk
spacy
```

## Setup Instructions

1. Clone the repository or download the project files.
2. Install the required dependencies using the `requirements.txt` file.
3. Download the `en_core_web_lg` model for spaCy:

   ```bash
   python -m spacy download en_core_web_lg
   ```

4. Open the Jupyter notebook `Credit card fraud detection.ipynb`.
5. Execute the notebook step-by-step to preprocess the data, train the models, and evaluate their performance.

## Methodology

### Data Preprocessing

- Data cleaning and handling missing values.
- Scaling the features using `StandardScaler` for better performance.
- Balancing the dataset using SMOTE (Synthetic Minority Over-sampling Technique).

### Model Building

The following machine learning models are used:

1. **Logistic Regression**
2. **Random Forest Classifier**
3. **Support Vector Machine (SVM)**

The models are evaluated based on the following metrics:
- Accuracy
- Precision
- Recall
- F1-score

### Model Evaluation

- Confusion matrix is plotted for each model.
- The models are compared to select the best-performing one for fraud detection.

## Usage

Run the notebook to:

1. Preprocess the credit card transaction data.
2. Train and evaluate different classification models.
3. Obtain predictions on test data and visualize performance using confusion matrices.

## Results

The performance of each model is evaluated using common classification metrics, and the confusion matrices are plotted for visual assessment. The model with the highest recall and precision is considered for deployment in real-world fraud detection scenarios.

## License

This project is open-source and available under the MIT License.

