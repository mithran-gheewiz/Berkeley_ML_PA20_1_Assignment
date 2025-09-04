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

Fig.1 Histogram L3_S29_F3470



