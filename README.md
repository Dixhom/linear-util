# Introduction
This is a helper class to perform feature engineering to datasets for linear models or neural networks. Unlike tree-based ones, those models need cumbersome feature engineering like scaling, missing value imputing and categorical feature encoding (tree-based models only require label encoding, which is quite simple). I wanted to let users save those cumbersome efforts and spend more time on valuable things like creating models. 

# Example
```
train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')

lfe = LinearFeatureEngieering(
        linear_scaling = 'standardize',
        nonlinear_scaling = 'log',
        num_features = ['num_feature1', 'num_feature2'],
        categorical_features = ['categorical_feature1', 'categorical_feature2'],
        fillna = None,
        fillna_fixed_value = None,
        fillna_category = None,
        fillna_bayesian_count = None,
        fillna_bayesian_value = None,
        clipping = None,
        ranking = None, 
        categorical_encoding = 'onehot')
lfe.fit(train)
train = lfe.transform(train)
test = lfe.transform(test)
```
