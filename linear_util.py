import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler, PowerTransformer, QuantileTransformer, OneHotEncoder, LabelEncoder
from sklearn.feature_extraction import FeatureHasher

from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split

from scipy import stats, sparse


class LinearFeatureEngieeringHelper:
    def __init__(
        self,
        feat = None,
        linear_scaling = None, # standardize, normalize
        nonlinear_scaling = None, # log, log1p, logabs, box-cox, yeo-johnson
        num_features = None,
        categorical_features = None,
        fillna = None, # fixed_value, mean, median, categorical_mean, bayesian_average
        fillna_fixed_value = None,
        fillna_category = None,
        fillna_bayesian_count = 10,
        fillna_bayesian_value = None,
        clipping = None,
        ranking = None, # rank, rankgauss
        categorical_encoding = None, #
        ):
    
        self.linear_scaling = linear_scaling
        self.nonlinear_scaling = nonlinear_scaling
        self.num_features = num_features
        self.categorical_features = categorical_features
        self.fillna = fillna
        self.fillna_fixed_value = fillna_fixed_value
        self.fillna_category = fillna_category
        self.fillna_bayesian_count = fillna_bayesian_count
        self.fillna_bayesian_value = fillna_bayesian_value
        self.clipping = clipping
        self.ranking = ranking
        self.categorical_encoding = categorical_encoding

    def bayesian_average(self, xs, c, v):
        return (xs.sum() + c * v) / (len(xs) + c)

    def fit(self, df, feat):
        if feat in self.num_features:
            # linear scaling
            if self.linear_scaling == 'standardize':
                self.standard_scaler = StandardScaler()
                self.standard_scaler.fit(df[feat].values.reshape(-1,1))
            elif self.linear_scaling == 'normalize':
                self.minmax_scaler = MinMaxScaler()
                self.minmax_scaler.fit(df[feat].values.reshape(-1,1))
            elif self.linear_scaling is None:
                pass
            else:
                raise ValueError(f'linear scaling option invalid: {self.linear_scaling}')
            
            # non linear scaling
            if self.nonlinear_scaling == 'log':
                pass
            elif self.nonlinear_scaling == 'log1p':
                pass
            elif self.nonlinear_scaling == 'logabs':
                pass
            elif self.nonlinear_scaling == 'box-cox':
                self.boxcox_transformer = PowerTransformer(method='box-cox')
                self.boxcox_transformer.fit(df[feat].values.reshape(-1,1))
            elif self.nonlinear_scaling == 'yeo-johnson':
                self.yeo_transfomer = PowerTransformer(method='yeo-johnson')
                self.yeo_transfomer.fit(df[feat].values.reshape(-1,1))
            elif self.nonlinear_scaling is None:
                pass
            else:
                raise ValueError(f'nonlinear scaling option invalid: {self.nonlinear_scaling}')
            
            # missing value imputation
            if self.fillna == 'fixed_value':
                pass
            elif self.fillna == 'mean':
                self.fillna_mean = df[feat].mean()
            elif self.fillna == 'median':
                self.fillna_median = df[feat].median()
            elif self.fillna == 'categorical_mean':
                # fill by the mean of values grouped by a category
                # the mean of values grouped by a category
                self.fillna_category_mean = df.groupby(self.fillna_category)[feat].mean().to_dict()
            elif self.fillna == 'bayesian_average':
                # fill by the mean of values grouped by a category
                # if a level in a category has too few data, its mean is not very trustworthy
                # so it's hard to use it to impute missing values.

                # if None use the global average
                if self.fillna_bayesian_value is None:
                    self.fillna_bayesian_value = df[feat].mean()

                # the mean of values grouped by a category
                self.fillna_bayesian_mean = df.groupby(self.fillna_category)[feat].apply(lambda x: self.bayesian_average(x, self.fillna_bayesian_count, self.fillna_bayesian_value)).to_dict()
            elif self.fillna is None:
                pass
            else:
                raise ValueError(f'missing value impuration option invalid: {self.fillna}')
                
            if self.clipping:
                if self.clipping > 1 or self.clipping < 0:
                    raise ValueError(f'clipping ({self.clipping})should be between 0 and 1.')
                self.clipping_lower = df[feat].quantile(self.clipping)
                self.clipping_upper = df[feat].quantile(1-self.clipping)
  
            if self.ranking == 'rankgauss':
                self.quantile_transformer = QuantileTransformer(n_quantiles=100, random_state=0, output_distribution='normal')
                self.quantile_transformer.fit(df[feat].values.reshape(-1,1))
        
        # categorical features
        elif feat in self.categorical_features:
            if self.categorical_encoding == 'onehot':
                self.ohe = OneHotEncoder(sparse=False)
                self.ohe.fit(df[feat].values.reshape(-1,1))

            elif self.categorical_encoding == 'label':
                self.le = LabelEncoder()
                self.le.fit(df[feat])
                
            elif self.categorical_encoding == 'feature_hashing':
                self.fh = FeatureHasher(n_features=10, input_type='string')
                self.fh.fit(df[feat].values.reshape(-1,1))
                
            elif self.categorical_encoding == 'frequency_encoding':
                cnt = df[feat].value_counts()
                self.frequency_encoding = cnt
                
            elif self.categorical_encoding is None:
                pass
            else:
                raise ValueError(f'categorical feature engineering option invalid {self.categorical_encoding}')
        else:
            raise ValueError(f'feature not in numeracal and categorical features: {feat}')
            
    def transform(self, df, feat):
        df = df.copy()
        fe_list = []
        
        if feat in self.num_features:
            # linear scaling
            if self.linear_scaling == 'standardize':
                sc_result = self.standard_scaler.transform(df[feat].values.reshape(-1,1))
                sc_result = pd.DataFrame({f'{feat}_sc': sc_result.ravel()})
                fe_list.append(sc_result)
            elif self.linear_scaling == 'normalize':
                mm_result = self.minmax_scaler.transform(df[feat].values.reshape(-1,1))
                mm_result = pd.DataFrame({f'{feat}_mm': mm_result.ravel()})
                fe_list.append(mm_result)
            elif self.linear_scaling is None:
                pass
            else:
                raise ValueError(f'linear scaling option invalid: {self.linear_scaling}')
            
            # non linear scaling
            if self.nonlinear_scaling == 'log':
                df[f'{feat}_log'] = np.log(df[feat])
            elif self.nonlinear_scaling == 'log1p':
                df[f'{feat}_log1p'] = np.log1p(df[feat])
            elif self.nonlinear_scaling == 'logabs':
                df[f'{feat}_logabs'] = np.sign(df[feat]) * np.log(np.abs(df[feat]))
            elif self.nonlinear_scaling == 'box-cox':
                bc_result = self.boxcox_transformer.transform(df[feat].values.reshape(-1,1))
                bc_result = pd.DataFrame(bc_result.ravel(), columns=[f'{feat}_bc'])
                fe_list.append(bc_result)
            elif self.nonlinear_scaling == 'yeo-johnson':
                yj_result = self.yeo_transfomer.transform(df[feat].values.reshape(-1,1))
                yj_result = pd.DataFrame(yj_result.ravel(), columns=[f'{feat}_yj'])
                fe_list.append(yj_result)
            elif self.nonlinear_scaling is None:
                pass
            else:
                raise ValueError(f'nonlinear scaling option invalid: {self.nonlinear_scaling}')
            
            # missing value imputation
            if self.fillna == 'fixed_value':
                fn_result = df[feat].fillna(value=self.fillna_fixed_value)
                fn_result.name = f'{feat}_fillna_fixed'
                fe_list.append(fn_result)
            elif self.fillna == 'mean':
                fm_result = df[feat].fillna(self.fillna_mean)
                fm_result.name = f'{feat}_fillna_mean'
                fe_list.append(fm_result)
            elif self.fillna == 'median':
                fm_result = df[feat].fillna(self.fillna_median)
                fm_result.name = f'{feat}_fillna_median'
                fe_list.append(fm_result)
            elif self.fillna == 'categorical_mean':
                # fill the nans
                fc_result = df[feat].copy()
                fc_result[fc_result.isnull()] = df[self.fillna_category][df[feat].isnull()].replace(self.fillna_category_mean)
                fc_result.name = f'{feat}_fillna_cat'
                fe_list.append(fc_result)
            elif self.fillna == 'bayesian_average':
                # fill by the mean of values grouped by a category
                # if a level in a category has too few data, its mean is not very trustworthy
                # so it's hard to use it to impute missing values.

                # fill the nans
                fb_result = df[feat].copy()
                fb_result[fb_result.isnull()] = df[self.fillna_category][df[feat].isnull()].replace(self.fillna_bayesian_mean)
                fb_result.name = f'{feat}_fillna_bayesian'
                fe_list.append(fb_result)
            elif self.fillna is None:
                pass
            else:
                raise ValueError(f'missing value imputation option invalid: {self.fillna}')
                
            if self.clipping:
                df[feat] = df[feat].clip(self.clipping_lower, self.clipping_upper)
                
            if self.ranking == 'rank':
                # if divided by len, the whole values will scaled to between 0 and 1. 
                # The handling will be easier. 
                
                rank_result = df[feat].rank() / len(df[feat])
                rank_result.name = f'{feat}_rank'
                fe_list.append(rank_result)
            elif self.ranking == 'rankgauss':
                rg_result = self.quantile_transformer.transform(df[feat].values.reshape(-1,1))
                rg_result = pd.DataFrame(rg_result, columns=[f'{feat}_rankgauss'])
                fe_list.append(rg_result)
            elif self.ranking is None:
                pass
            else:
                raise ValueError(f'ranking option invalid: {self.ranking}')
                
        
        # categorical features
        elif feat in self.categorical_features:
            if self.categorical_encoding == 'onehot':
                ohe_result = self.ohe.fit_transform(df[feat].values.reshape(-1,1))
                n_unique = df[feat].nunique()
                columns = [f'{feat}_ohe_{v}' for v in range(n_unique)]
                ohe_result = pd.DataFrame(ohe_result, columns=columns)
                fe_list.append(ohe_result)
                
            elif self.categorical_encoding == 'label':
                le_result = self.le.transform(df[feat])
                le_result = pd.DataFrame(le_result, columns=[f'{feat}_le'])
                fe_list.append(le_result)
                
            elif self.categorical_encoding == 'feature_hashing':
                fh_result = self.fh.transform(df[feat].values.reshape(-1,1))
                cols = [f'{feat}_fh_{i}' for i in range(fh.n_features)]
                fh_result = pd.DataFrame.sparse.from_spmatrix(fh_result, columns=cols)
                fe_list.append(fh_result)
            elif self.categorical_encoding == 'frequency_encoding':
                freq_result = df[feat].map(self.frequency_encoding)
                freq_result.name = f'{feat}_freq_enc'
                fe_list.append(freq_result)
            elif self.categorical_encoding is None:
                pass
            else:
                raise ValueError(f'categorical encoding option invalid: {self.categorical_encoding}')
                
        else:
            raise ValueError(f'feature not in numeracal and categorical features: {feat}')
        
        return pd.concat([df] + fe_list, axis=1)
        
class LinearFeatureEngieering:
    def __init__(
        self,
        linear_scaling = None, # standardize, normalize
        nonlinear_scaling = None, # log, log1p, logabs, box-cox, yeo-johnson
        num_features = [],
        categorical_features = [],
        fillna = 'fixed_value', # mean, median, categorical_mean, bayesian_average
        fillna_fixed_value = None,
        fillna_category = None,
        fillna_bayesian_count = 10,
        fillna_bayesian_value = None,
        clipping = None,
        ranking = None,
        categorical_encoding = None,
        ):
    
        self.all_features = num_features + categorical_features
        
        keywords = dict(
            linear_scaling = linear_scaling,
            nonlinear_scaling = nonlinear_scaling,
            num_features = num_features,
            categorical_features = categorical_features,
            fillna = fillna,
            fillna_fixed_value = fillna_fixed_value,
            fillna_category = fillna_category,
            fillna_bayesian_count = fillna_bayesian_count,
            fillna_bayesian_value = fillna_bayesian_value,
            clipping = clipping,
            ranking = ranking,
            categorical_encoding = categorical_encoding,
        )
        
        self.transformers = {f: LinearFeatureEngieeringHelper(feat=f, **keywords) for f in self.all_features}

    def fit(self, df):
        for feat in self.all_features:
            self.transformers[feat].fit(df, feat)
            
    def transform(self, df):
        for feat in self.all_features:
            df = self.transformers[feat].transform(df, feat)
        return df
            
    def fit_transform(self, df):
        self.fit(df)
        return self.transform(df)
    