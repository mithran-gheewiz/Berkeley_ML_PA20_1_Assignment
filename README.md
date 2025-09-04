# Berkeley_ML_PA20_1_Assignment

## Link to Jupyter Notebook: 

### Data Checks
The data set is from the Bosch Kaggle competition. The data is sourced from Kaggle. https://www.kaggle.com/competitions/bosch-production-line-performance/overview

The dataset is huge containing three types of feature data: numerical, categorical, date stamps and the labels indicating the part as good or bad. The training data has 1184687 samples and the learned model will be used to predict on a test dataset containing 1183748 samples. There are 968 numerical features, 2140 categorical features and 1156 date features. I used a limited data set of randomized 100,000 rows from the training datasets and randomized 30,000 rows from the test datasets.

Using random sampling, I was able to get the following rows from the training data sets:  
train_numerical: (99922, 970), First ID: 49
train_categorical: (99922, 2141), First ID: 49
train_date: (99922, 1157), First ID: 49

The data checks shows that there is a large number of missing values in the sampled 100,000 data frame for all three files (numerical, categorical, and date). This will pose a significant issue for logistic regression which is sensitive to NaN or missing values. Must drop rows that have lots of NaNs, and/or use PCA and/or L1 regression that will reduce factors that are not important to zero. Here is an example of the average number of missing values:
Average NaNs per row numerical: 784.98
Average NaNs per row categorical: 2082.95
Average NaNs per row date: 950.86

Similarly, I also randomly sampled 30,000 rows from the test data sets with the following results: 
test_numerical: (30001, 969), First ID: 19
test_categorical: (30001, 2141), First ID: 19
test_date: (30001, 1157), First ID: 19

### Exploratory data analysis (EDA)
For EDA, I focused on the following items:
- Correlation matrix
- Univariate analysis - density, count plots, histograms, summary statistics
- Bivariate analysis - Correlation matrix, scatterplots, box plots
- Multivariate analysis - map different aspects to a chart

Based on the tables, I noticed that certain stations had a higher frequency of parts passing through that station. For example, station L0_S2_F33 (L0_S2_F33 means the Feature 33 measured at Station 2 on Assembly Line 0) has a frequency of 25. There are many stations, where parts did not pass through at all and this is tabulated with missing_pct at 1.0 (100% missing). These will be dropped before fitting for the logistic regression.

                      Table 1. Count and Frequency at different stations
                      
<img width="343" height="429" alt="image" src="https://github.com/user-attachments/assets/1eae27b8-3223-495c-8ba3-0e1398b0ce6f" />

Univariate analysis – numeric (histograms & density/KDE). I picked a manageable subset to visualize (top 12 by variance or a manual list).

<img width="1253" height="685" alt="image" src="https://github.com/user-attachments/assets/ddf168a5-cbcf-4e9b-868b-41a50a8bd16f" />

                                      Fig.1 Histogram L3_S29_F3470 (one example)

The figure above shows the only two different numeric features are measured at that station. There are examples of multiple measurements being made at a particular station which is shown in Figure 2 below.

<img width="1247" height="688" alt="image" src="https://github.com/user-attachments/assets/9b1611b4-e079-4185-a909-f801cefbee4c" />

                                      Fig.2 Histogram L0_S7_F138 (one example)

I also made some density plots on the individual features. An example is shown below in figure 3. 

<img width="1272" height="720" alt="image" src="https://github.com/user-attachments/assets/a94d1983-d3fb-4544-8115-f3b4a54c7cd6" />

                                      Fig. 3. Density plot L1_S25_F2623

Because the data are is so sparse with many missing values, I obtained many figures that shows completely missing values for a particular station.

For the corelation matrix, the heatmap shows strong correlation between some features in Line1 Station 24 for features 766, 780, 785 and 775 for the high variance. In the top 25, there are high negative correlation for F2906 and F2925 all from Line1 and Station 25 suggesting some inverse correlation.

<img width="901" height="788" alt="image" src="https://github.com/user-attachments/assets/fababf2e-20d2-4047-8aee-c9a21e098809" />

                                      Fig. 4. Correlation heat map for the top 25 high variance numeric features


### Preprocessing
For preprocessing, I used the following methodology:
- Categorical to numerical for both the test and train data: One hot encoder (OHE)
- Numerical data: imputation, standard scaling (since the data sets are separate don't have to worry to do it after the test train split. Can perform the standard scaling on each test and train data set)
- Compare against minimum – maximum scaling
All-NaN numeric columns are dropped before fitting to avoid SimpleImputer errors.
Categorical imputation uses a constant label ("MISSING") so all-missing columns still produce a single OHE column (rather than erroring).
By setting handle_unknown='ignore' allows the test set have unseen categories without crashing.
Two preprocessors (Standard vs Min-Max) are trained independently to compare downstream model performance easily.
I fit on train, apply to test: both imputers, scalers, and the OHE learn from train only, preventing leakage.
  
The transformed matrices are shown below: 
X_train_std: shape=(99922, 4908), sparse=False, density=n/a
X_test_std : shape=(30001, 4908), sparse=False, density=n/a
X_train_mm : shape=(99922, 4908), sparse=False, density=n/a
X_test_mm  : shape=(30001, 4908), sparse=False, density=n/a

### Feature Selection 
I used principal component analysis (PCA) to identify features that are important for the model

<img width="943" height="429" alt="image" src="https://github.com/user-attachments/assets/2d2d06bc-cea3-45d2-aa93-a4c5cb4ff313" />

                                    Fig. 5. Scree: EVR per Component 

The first few components explain the most variance (PC1 ≈ 0.03 = 3%).
After ~10–20 components, the curve flattens out, meaning later components each explain very little variance (<0.5% each).
I make sense to limit only the first few components at the elbow before the curve flattens out.  

<img width="937" height="430" alt="image" src="https://github.com/user-attachments/assets/338f9280-d8cf-44a9-b0dd-4eee23936e16" />

                                    Fig. 6. Cumulative Explained Variance 

From Fig. 6, With ~300 components, the cumulative explained variance reaches only ~0.77 (77%).
This means that even after keeping the first 300 PCs, the model does not capture 95% of the variance (EVR >= 95%)

### Modeling 
I use the following baseline models in the analysis: 
- Logistic regression model - L1 regularization
- Decision tree model (Tree depth (n=5))
- Hyper parameter tuning for both logistic regression and decision tree

When I first started out, the baseline model did not converge. Then, I performed dimensionality reduction before the classifier (often best for OHE-heavy data) and reran the program and its still running after 12 hours without completing the kernal. 
I determined that the bottlenecks are: (1) refitting OHE+SVD inside every CV fold, (2) too many candidates/folds, and (3) using a heavy solver on high-dim data.
To make it run faster, I did the following:
Tune on a stratified subset (30k rows), then refit best params on the full 100k.
Use HalvingRandomSearchCV (successive halving) with 3-fold CV.
Shrink SVD to ~50 comps and n_iter=2.
Use L1 + liblinear (binary) after SVD (way faster than saga here).
For the tree, compress the categorical branch with OHE → SVD inside the ColumnTransformer (dramatically fewer features).  
                                      
