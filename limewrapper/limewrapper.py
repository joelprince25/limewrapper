import h2o
import pandas as pd
import numpy as np
import lime
import lime.lime_tabular
from sklearn.preprocessing import LabelEncoder

class LimeWrapperH2O():
    def __init__(self,
                 model,
                 df_train,
                 df_test,
                 col_type_dict,
                 features,
                 target):

        self.model = model
        self.df_train = df_train.copy()
        self.df_test = df_test.copy()
        self.col_type_dict = col_type_dict
        self.col_type_dict_x = {key : value for key, value in self.col_type_dict.items() if key in features}
        self.features = features
        self.target = target

        self.df_train = self.df_train[features + [target]]
        self.df_test = self.df_test[features]
        self.convert_types(self.df_train)
        self.convert_types(self.df_test)

        self.target_col = self.df_train[self.target]
        self.df_train.drop(target, axis = 1, inplace = True)
        
        self.categorical_columns = [col
                                    for col
                                    in self.df_train.columns
                                    if (col_type_dict[col] == 'enum')]
        self.numerical_columns = [col
                                  for col
                                  in self.df_train.columns
                                  if col_type_dict[col] == 'real']
        self.categorical_feat_index =  [list(self.df_train.columns.values).index(col)
                                        for col
                                        in self.df_train.columns.values
                                        if col
                                        in self.categorical_columns]
        self.impute(self.df_train)
        self.impute(self.df_test)
        self.preprocess()
        self.model_refit()


    def convert_types(self, df):
        for col in df.columns.values:
            if self.col_type_dict[col] == 'enum':
                df[col] = df[col].astype(str)
            else:
                df[col] = df[col].astype(float)


    def impute(self, df):
        for col in df.columns:
            if col in self.categorical_columns:
                df[col].fillna('-999', inplace = True)
            else:
                df[col].fillna(df[col].median(), inplace = True)

    def preprocess(self):
        if self.col_type_dict[self.target] == 'enum':
            le = LabelEncoder()
            self.labels = self.target_col.values
            le.fit(self.labels)
            self.labels = le.transform(self.labels)
            self.class_names = le.classes_
        else:
            self.labels = self.target_col.values

        self.train = self.df_train.values
        self.test = self.df_test.values

        self.categorical_names = {}
        for feature in self.categorical_feat_index:
            le = LabelEncoder()
            le.fit(np.concatenate((self.train[:,feature], self.test[:,feature]), axis = 0))
            self.train[:, feature] = le.transform(self.train[:, feature])
            self.test[:, feature] = le.transform(self.test[:, feature])
            self.categorical_names[feature] = le.classes_

        self.train = self.train.astype(float)
        self.test = self.test.astype(float)

        self.train_h2o_df = h2o.H2OFrame(self.train, column_names = self.features, column_types = self.col_type_dict_x)
        self.train_h2o_df[self.target] = h2o.H2OFrame(self.labels)

        if self.col_type_dict[self.target] == 'enum':
            self.train_h2o_df[self.target] = self.train_h2o_df[self.target].asfactor()

        self.test_h2o_df = h2o.H2OFrame(self.test, column_names = self.features, column_types = self.col_type_dict_x)

        #for feature in self.categorical_feat_index:
            #self.train_h2o_df[feature] = self.train_h2o_df[feature].asfactor()
            #self.test_h2o_df[feature] = self.test_h2o_df[feature].asfactor()

    def model_refit(self):
        if self.categorical_columns:
            self.model.train(x = self.features, y = self.target, training_frame = self.train_h2o_df)

    def predict_proba(self,this_array):
        # If we have just 1 row of data we need to reshape it
        shape_tuple = np.shape(this_array)
        if len(shape_tuple) == 1:
            this_array = this_array.reshape(1, -1)

        # We convert the numpy array that Lime sends to a pandas dataframe and
        # convert the pandas dataframe to an h2o frame
        self.pandas_df = pd.DataFrame(data = this_array,columns = self.features)
        self.h2o_df = h2o.H2OFrame(self.pandas_df)

        # Predict with the h2o drf
        self.predictions = self.model.predict(self.h2o_df).as_data_frame()
        # the first column is the class labels, the rest are probabilities for each class
        self.predictions = self.predictions.iloc[:,1:].as_matrix()
        return self.predictions

    def predict(self,this_array):
        # If we have just 1 row of data we need to reshape it
        shape_tuple = np.shape(this_array)
        if len(shape_tuple) == 1:
            this_array = this_array.reshape(1, -1)

        # We convert the numpy array that Lime sends to a pandas dataframe and
        # convert the pandas dataframe to an h2o frame
        self.pandas_df = pd.DataFrame(data = this_array,columns = self.features)
        self.h2o_df = h2o.H2OFrame(self.pandas_df)

        # Predict with the h2o drf
        self.predictions = self.model.predict(self.h2o_df).as_data_frame().values.flatten()
        return self.predictions

    def create_explainer(self,
                         mode,
                         kernel_width = None,
                         feature_selection = 'auto',
                         verbose = False,
                         random_state = None,
                         **kwargs):
        if mode == 'classification':
            self.mode = mode
            self.explainer = lime.lime_tabular.LimeTabularExplainer(self.train,
                                                                    mode = mode,
                                                                    feature_names = self.features,
                                                                    class_names = self.class_names,
                                                                    categorical_features = self.categorical_feat_index,
                                                                    categorical_names = self.categorical_names,
                                                                    kernel_width = kernel_width,
                                                                    feature_selection = feature_selection,
                                                                    verbose = verbose,
                                                                    random_state = random_state,
                                                                    **kwargs)
        elif mode == 'regression':
            self.mode = mode
            self.explainer = lime.lime_tabular.LimeTabularExplainer(self.train,
                                                                    mode = mode,
                                                                    feature_names = self.features,
                                                                    categorical_features = self.categorical_feat_index,
                                                                    categorical_names = self.categorical_names,
                                                                    kernel_width = kernel_width,
                                                                    feature_selection = feature_selection,
                                                                    verbose = verbose,
                                                                    random_state = random_state,
                                                                    **kwargs)
        else:
            print ("mode must be either classification or regression")

    def explain(self, i, num_features = 10, **kwargs):
        if self.mode == 'classification':
            self.exp = self.explainer.explain_instance(self.test[i], self.predict_proba, num_features = num_features, **kwargs)
        elif self.mode == 'regression':
            self.exp = self.explainer.explain_instance(self.test[i], self.predict, num_features = num_features, **kwargs)
        self.exp.show_in_notebook()
