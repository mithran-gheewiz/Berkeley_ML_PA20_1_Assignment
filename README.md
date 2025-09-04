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
