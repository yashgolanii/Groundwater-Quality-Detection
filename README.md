# Ground Water Quality Classification

This project aims to classify ground water quality using various machine learning models. The classification is based on a dataset containing ground water quality parameters for the years 2018, 2019, and 2020. Multiple models such as K-Nearest Neighbors, Logistic Regression, Support Vector Machine, Decision Tree, and Random Forest Classifier are used for classification tasks.

## Dataset

The dataset consists of ground water quality data for three years:
- `ground_water_quality_2018_post.csv`
- `ground_water_quality_2019_post.csv`
- `ground_water_quality_2020_post.csv`

### Data Preprocessing

The preprocessing steps include:
- Handling missing values by filling them with mean values.
- Renaming certain columns for consistency.
- Concatenating the datasets from three years into one.
- Transforming features using log transformation for better model performance.
- Encoding categorical variables into numerical values.

### Exploratory Data Analysis (EDA)

EDA is performed to understand the distribution and relationships of various parameters in the dataset. Scatter plots and histograms are used to visualize the data.

### Feature Engineering

The following features are selected for modeling:
- Latitude (`lat_gis`)
- Longitude (`long_gis`)
- Ground Water Level (`gwl`)
- pH
- Electrical Conductivity (`E.C`)
- Total Dissolved Solids (`TDS`)
- Carbonates (`CO3`)
- Bicarbonates (`HCO3`)
- Chlorides (`Cl`)
- Fluorides (`F`)
- Nitrates (`NO3`)
- Sulfates (`SO4`)
- Sodium (`Na`)
- Potassium (`K`)
- Calcium (`Ca`)
- Magnesium (`Mg`)
- Total Hardness (`T.H`)
- Sodium Adsorption Ratio (`SAR`)
- Residual Sodium Carbonate (`RSC`)

### Classification Models

#### 1. K-Nearest Neighbors (KNN)

Two KNN models are trained:
- Model 1: Classifies water quality based on multiple classes.
- Model 2: Classifies water quality into three classes: PS, US, MR.

#### 2. Logistic Regression

Two Logistic Regression models are trained:
- Model 1: For multi-class classification.
- Model 2: For three-class classification (PS, US, MR).

#### 3. Support Vector Machine (SVM)

Four different kernels are used for SVM:
- Linear
- Polynomial
- Radial Basis Function (RBF)
- Sigmoid

Grid Search with cross-validation is performed to find the best hyperparameters for each kernel.

#### 4. Decision Tree Classifier

Two Decision Tree models are trained:
- Model 1: For multi-class classification.
- Model 2: For three-class classification (PS, US, MR).

#### 5. Random Forest Classifier

Two Random Forest models are trained:
- Model 1: For multi-class classification with hyperparameter tuning.
- Model 2: For three-class classification (PS, US, MR) with hyperparameter tuning.

## Usage

### Prerequisites

- Python 3.x
- Libraries: pandas, numpy, matplotlib, seaborn, scikit-learn, google.colab

### Steps to Run

1. Mount Google Drive:
    ```python
    from google.colab import drive
    drive.mount('/content/drive')
    ```

2. Load the datasets:
    ```python
    df_1 = pd.read_csv('/content/drive/MyDrive/project/ground_water_quality_2018_post.csv')
    df_2 = pd.read_csv('/content/drive/MyDrive/project/ground_water_quality_2019_post.csv')
    df_3 = pd.read_csv('/content/drive/MyDrive/project/ground_water_quality_2020_post.csv')
    ```

3. Preprocess the data as shown in the script.

4. Perform EDA using the provided visualizations.

5. Train the models:
    - KNN
    - Logistic Regression
    - SVM
    - Decision Tree
    - Random Forest

6. Evaluate the models using accuracy, confusion matrix, and classification report.

### Model Evaluation

The models are evaluated using metrics such as accuracy, precision, recall, F1-score, and confusion matrix. The best parameters for each model are determined using Grid Search with cross-validation.

## Results

The results of the models are visualized using heatmaps for confusion matrices and plots for accuracy scores. The detailed classification reports provide insights into the performance of each model.

## Conclusion

This project demonstrates the application of various machine learning models for classifying ground water quality. The Random Forest model with 33 estimators showed the best performance for multi-class classification, while the SVM with the RBF kernel performed well for three-class classification.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- Special thanks to the Google Colab platform for providing the environment to run this project.
- Data sourced from the ground water quality monitoring agencies.

