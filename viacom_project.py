import pandas as pd
import glob
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import *
from sklearn.ensemble import RandomForestRegressor

cpm_estim = pd.read_csv('D:/NEU-2019-Social-demo-targeting/cpm_estimates_25Jan19.csv')
cpm_estim.dtypes

cpm_estim['age-group'] = cpm_estim['age_min'].astype(str) + '-' + cpm_estim['age_max']

def f(x):
  if (x['male'] == 1) and (x['female'] == 1): return 'Both'
  elif x['female'] == 1: return 'female'
  elif x['male'] == 1 : return 'Male'
  else: return 'NAN'

cpm_estim['Gender']= cpm_estim.apply(f, axis=1)
# cpm_estim.to_csv("cpm_file.csv")

types = cpm_estim.dtypes

num_des = cpm_estim.describe(include=[np.number])
cat_desc = cpm_estim.describe(include=[np.object])

cpm_estim['age-group'].value_counts()

# getting dummy variable
cpm_dumm = pd.get_dummies(cpm_estim, prefix="", columns=['Gender'],drop_first = True)

# Replaced age group with values
set_id = {'13-17':0, '18-24':1, '25-34':2, '35-44':3,
          '45-54':4, '18-34':5, '18-44':6, '18-49':7,
          '18-54':8, '18-65+':9, '25-54':10, '34-44':11, '44-54':12}

cpm_dumm.rename\
  (columns={'age-group':'age_group','_Male':'Male','_female':'Female'},
   inplace=True)

cpm_dumm['age_group'] = cpm_dumm.age_group.replace(set_id)

# Extracting the required columns
new = ['Male', 'Female', 'age_group', 'date', 'hID', 'cpm']
cpm_fin = cpm_dumm[new]
cpm_fin =pd.DataFrame(cpm_fin)

# modifying the date column
# suppressing the '-' in date column and convert it into int64
cpm_fin['date']=cpm_fin['date'].str.replace("-", "").astype(np.int64)
cpm_fin.dtypes

# Regression
# Correlation using seaborn

corr = cpm_fin.corr()

# plot the heatmap
sns.heatmap(corr, xticklabels=corr.columns, yticklabels=corr.columns)

X = cpm_fin.iloc[:, :-1].values
Y = cpm_fin.iloc[:, 5].values

X_train, X_valid, Y_train, Y_valid = train_test_split(X,Y, test_size = 0.2, random_state= 42)

# Fitting the model
mod_fit = LinearRegression()
mod_fit.fit(X_train, Y_train)


# Predicting the validation set results
Y_valid_Pred = mod_fit.predict(X_valid)

# calculating the coefficients
print(mod_fit.coef_)

# Calculating the intercept
print(mod_fit.intercept_)

# Calculating the R square value
r2_score(Y_valid,Y_valid_Pred)

# calculating the mean square error
msq = mean_squared_error(Y_valid,Y_valid_Pred)

# We will also use random forest for validation
mod_RF = RandomForestRegressor(max_depth=10, random_state=42)
mod_RF

# Train the model using the training sets
mod_RF.fit(X_train, Y_train)

# Make predictions using the testing set
y_pred_valid = mod_RF.predict(X_valid)

# calculating the coefficients
print(mod_fit.coef_)

# Calculating the intercept
print(mod_fit.intercept_)

# Calculating the R square value
rf_r2score = r2_score(Y_valid, y_pred_valid)

# calculating the mean square error
msq_error = mean_squared_error(Y_valid,Y_valid_Pred)


# as per our analysis we will use random forest regression model used above for
# predicting the unknown CPM values of page level HID

# Aim 1:Concat all the files at a once
# Reading files as a whole
files = glob.glob('D:/NEU-2019-Social-demo-targeting/page_level_data/*.csv')
combined_csv = pd.concat( [ pd.read_csv(f) for f in files ] )
combined_csv['hID'].nunique()

combined_csv.name.nunique()
# Aim 2: Partitioning on the basis of names

ls = []
list = combined_csv.name.unique()
len(list)
for nam in list:
    p = combined_csv.loc[combined_csv['name'] == nam].sample(frac=0.30,random_state=42)
    ls.append(p)
pg_level = pd.concat(ls)

pg_types = pg_level.dtypes

num_des = pg_level.describe(include=[np.number])
cat_desc = pg_level.describe(include=[np.object])


counts = pg_level.groupby('name').nunique()

pg_level.to_csv('final_level_data.csv')

pg_level['hID'].nunique()

pg_level['date'] = pg_level['date'].str.replace("-","").astype(np.int64)
pg = pg_level

pg = pg.loc[pg['name'] == 'page_impressions_by_age_gender_unique']
pg_organic = pg_level.loc[pg_level['name'] == 'page_impressions_by_age_gender_unique']
pg_paid = pg_level.loc[pg_level['name'] == 'page_impressions_paid']

pg_level.name.nunique()
pg['gender']= pg['metric'].str.split('.').str[0]
pg['age_group'] = pg['metric'].str[2:]


pg.to_csv("final_data.csv")

pg.count()


col = [0,2,3,4]
pg = pg.drop([pg.columns[0], pg.columns[2], pg.columns[3], pg.columns[4]],axis=1)
pg['age_group'].value_counts()

pg = pg.drop(pg[pg.age_group == '65+'].index)

pg.dtypes

age_id = {'13-17': 0,
          '18-24': 1,
          '25-34': 2,
          '35-44': 3,
          '45-54': 4,
          '55-64': 5,
          }
pg['age_group'] = pg.age_group.replace(age_id)

pg = pd.DataFrame(pg)
pg = pd.get_dummies(pg, prefix="", columns=['gender'])
pg = pg.drop('_U',1)

pg.rename(columns={'_M': 'Male', '_F': 'Female'}, inplace=True)
pg.dtypes

# pg['date'] = pg['date'].str.replace("-","").astype(np.int64)
# Predicting the cpm values for the 593 HID's

pg_level_test = pg

# Using random forest model

pg_level_cpm = mod_RF.predict(pg_level_test)
pg['cpm_pred'] = pg_level_cpm

combined_csv['name'].nunique()


# Finding the impressions_paid and impressions_unpaid

paid = pd.merge(pg, pg_paid, left_on='hID',right_on='hID')

unpaid = pd.merge(pg, pg_organic,left_on='hID',right_on='hID')

col = ['hID', 'name', 'Male', 'Female', 'age_group', 'value', 'cpm_pred']
paid = paid[col]
unpaid = unpaid[col]

paid = paid.replace('', np.NaN)
paid = paid.dropna(how='any')

unpaid = unpaid.replace('', np.NaN)
unpaid = unpaid.dropna(how='any')
age_id = {0:'13-17',
          1:'18-24',
          2:'25-34',
          3:'35-44',
          4:'45-54',
          5:'55-64',
          }

paid['age_group'] = paid.age_group.replace(age_id)
unpaid['age_group'] = unpaid.age_group.replace(age_id)
paid.head(50)



unpaid = pd.merge(pg, pg_organic,left_on='hID',right_on='hID')

