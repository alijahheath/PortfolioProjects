# Insurance Fraud Prediction Project

In this project, I build a predictive model to estimate whether an insurance claim is potentially fraudulent, using demographic and policy-related features. I start by exploring and cleaning the dataset, handling missing values, encoding categorical variables using one-hot encoding, and standardizing numeric features. I then train a logistic regression model and evaluate its performance using classification metrics such as precision, recall, F1-score, and accuracy. I also check for signs of overfitting by comparing training and test set performance.


```python
import pandas as pd
```


```python
df = pd.read_csv('C:/Users/lijah/Desktop/Python Project 2025/insurance_claims.csv')

```


```python
#Display info about the dataset, preview the first few rows of the dataset
df.info()
df.head()
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 1000 entries, 0 to 999
    Data columns (total 40 columns):
     #   Column                       Non-Null Count  Dtype  
    ---  ------                       --------------  -----  
     0   months_as_customer           1000 non-null   int64  
     1   age                          1000 non-null   int64  
     2   policy_number                1000 non-null   int64  
     3   policy_bind_date             1000 non-null   object 
     4   policy_state                 1000 non-null   object 
     5   policy_csl                   1000 non-null   object 
     6   policy_deductable            1000 non-null   int64  
     7   policy_annual_premium        1000 non-null   float64
     8   umbrella_limit               1000 non-null   int64  
     9   insured_zip                  1000 non-null   int64  
     10  insured_sex                  1000 non-null   object 
     11  insured_education_level      1000 non-null   object 
     12  insured_occupation           1000 non-null   object 
     13  insured_hobbies              1000 non-null   object 
     14  insured_relationship         1000 non-null   object 
     15  capital-gains                1000 non-null   int64  
     16  capital-loss                 1000 non-null   int64  
     17  incident_date                1000 non-null   object 
     18  incident_type                1000 non-null   object 
     19  collision_type               1000 non-null   object 
     20  incident_severity            1000 non-null   object 
     21  authorities_contacted        1000 non-null   object 
     22  incident_state               1000 non-null   object 
     23  incident_city                1000 non-null   object 
     24  incident_location            1000 non-null   object 
     25  incident_hour_of_the_day     1000 non-null   int64  
     26  number_of_vehicles_involved  1000 non-null   int64  
     27  property_damage              1000 non-null   object 
     28  bodily_injuries              1000 non-null   int64  
     29  witnesses                    1000 non-null   int64  
     30  police_report_available      1000 non-null   object 
     31  total_claim_amount           1000 non-null   int64  
     32  injury_claim                 1000 non-null   int64  
     33  property_claim               1000 non-null   int64  
     34  vehicle_claim                1000 non-null   int64  
     35  auto_make                    1000 non-null   object 
     36  auto_model                   1000 non-null   object 
     37  auto_year                    1000 non-null   int64  
     38  fraud_reported               1000 non-null   object 
     39  _c39                         0 non-null      float64
    dtypes: float64(2), int64(17), object(21)
    memory usage: 312.6+ KB
    




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>months_as_customer</th>
      <th>age</th>
      <th>policy_number</th>
      <th>policy_bind_date</th>
      <th>policy_state</th>
      <th>policy_csl</th>
      <th>policy_deductable</th>
      <th>policy_annual_premium</th>
      <th>umbrella_limit</th>
      <th>insured_zip</th>
      <th>...</th>
      <th>police_report_available</th>
      <th>total_claim_amount</th>
      <th>injury_claim</th>
      <th>property_claim</th>
      <th>vehicle_claim</th>
      <th>auto_make</th>
      <th>auto_model</th>
      <th>auto_year</th>
      <th>fraud_reported</th>
      <th>_c39</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>328</td>
      <td>48</td>
      <td>521585</td>
      <td>2014-10-17</td>
      <td>OH</td>
      <td>250/500</td>
      <td>1000</td>
      <td>1406.91</td>
      <td>0</td>
      <td>466132</td>
      <td>...</td>
      <td>YES</td>
      <td>71610</td>
      <td>6510</td>
      <td>13020</td>
      <td>52080</td>
      <td>Saab</td>
      <td>92x</td>
      <td>2004</td>
      <td>Y</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>1</th>
      <td>228</td>
      <td>42</td>
      <td>342868</td>
      <td>2006-06-27</td>
      <td>IN</td>
      <td>250/500</td>
      <td>2000</td>
      <td>1197.22</td>
      <td>5000000</td>
      <td>468176</td>
      <td>...</td>
      <td>?</td>
      <td>5070</td>
      <td>780</td>
      <td>780</td>
      <td>3510</td>
      <td>Mercedes</td>
      <td>E400</td>
      <td>2007</td>
      <td>Y</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>2</th>
      <td>134</td>
      <td>29</td>
      <td>687698</td>
      <td>2000-09-06</td>
      <td>OH</td>
      <td>100/300</td>
      <td>2000</td>
      <td>1413.14</td>
      <td>5000000</td>
      <td>430632</td>
      <td>...</td>
      <td>NO</td>
      <td>34650</td>
      <td>7700</td>
      <td>3850</td>
      <td>23100</td>
      <td>Dodge</td>
      <td>RAM</td>
      <td>2007</td>
      <td>N</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>3</th>
      <td>256</td>
      <td>41</td>
      <td>227811</td>
      <td>1990-05-25</td>
      <td>IL</td>
      <td>250/500</td>
      <td>2000</td>
      <td>1415.74</td>
      <td>6000000</td>
      <td>608117</td>
      <td>...</td>
      <td>NO</td>
      <td>63400</td>
      <td>6340</td>
      <td>6340</td>
      <td>50720</td>
      <td>Chevrolet</td>
      <td>Tahoe</td>
      <td>2014</td>
      <td>Y</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>4</th>
      <td>228</td>
      <td>44</td>
      <td>367455</td>
      <td>2014-06-06</td>
      <td>IL</td>
      <td>500/1000</td>
      <td>1000</td>
      <td>1583.91</td>
      <td>6000000</td>
      <td>610706</td>
      <td>...</td>
      <td>NO</td>
      <td>6500</td>
      <td>1300</td>
      <td>650</td>
      <td>4550</td>
      <td>Accura</td>
      <td>RSX</td>
      <td>2009</td>
      <td>N</td>
      <td>NaN</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 40 columns</p>
</div>




```python
#Check for missing values:
df.isnull().sum()

```




    months_as_customer                0
    age                               0
    policy_number                     0
    policy_bind_date                  0
    policy_state                      0
    policy_csl                        0
    policy_deductable                 0
    policy_annual_premium             0
    umbrella_limit                    0
    insured_zip                       0
    insured_sex                       0
    insured_education_level           0
    insured_occupation                0
    insured_hobbies                   0
    insured_relationship              0
    capital-gains                     0
    capital-loss                      0
    incident_date                     0
    incident_type                     0
    collision_type                    0
    incident_severity                 0
    authorities_contacted             0
    incident_state                    0
    incident_city                     0
    incident_location                 0
    incident_hour_of_the_day          0
    number_of_vehicles_involved       0
    property_damage                   0
    bodily_injuries                   0
    witnesses                         0
    police_report_available           0
    total_claim_amount                0
    injury_claim                      0
    property_claim                    0
    vehicle_claim                     0
    auto_make                         0
    auto_model                        0
    auto_year                         0
    fraud_reported                    0
    _c39                           1000
    dtype: int64




```python
#Check for duplicate rows
df.duplicated().sum()
```




    0




```python
#No duplicates found
#Display column names
df.columns
```




    Index(['months_as_customer', 'age', 'policy_number', 'policy_bind_date',
           'policy_state', 'policy_csl', 'policy_deductable',
           'policy_annual_premium', 'umbrella_limit', 'insured_zip', 'insured_sex',
           'insured_education_level', 'insured_occupation', 'insured_hobbies',
           'insured_relationship', 'capital-gains', 'capital-loss',
           'incident_date', 'incident_type', 'collision_type', 'incident_severity',
           'authorities_contacted', 'incident_state', 'incident_city',
           'incident_location', 'incident_hour_of_the_day',
           'number_of_vehicles_involved', 'property_damage', 'bodily_injuries',
           'witnesses', 'police_report_available', 'total_claim_amount',
           'injury_claim', 'property_claim', 'vehicle_claim', 'auto_make',
           'auto_model', 'auto_year', 'fraud_reported', '_c39'],
          dtype='object')




```python
#Summary statistics
df.describe()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>months_as_customer</th>
      <th>age</th>
      <th>policy_number</th>
      <th>policy_deductable</th>
      <th>policy_annual_premium</th>
      <th>umbrella_limit</th>
      <th>insured_zip</th>
      <th>capital-gains</th>
      <th>capital-loss</th>
      <th>incident_hour_of_the_day</th>
      <th>number_of_vehicles_involved</th>
      <th>bodily_injuries</th>
      <th>witnesses</th>
      <th>total_claim_amount</th>
      <th>injury_claim</th>
      <th>property_claim</th>
      <th>vehicle_claim</th>
      <th>auto_year</th>
      <th>_c39</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>1000.000000</td>
      <td>1000.000000</td>
      <td>1000.000000</td>
      <td>1000.000000</td>
      <td>1000.000000</td>
      <td>1.000000e+03</td>
      <td>1000.000000</td>
      <td>1000.000000</td>
      <td>1000.000000</td>
      <td>1000.000000</td>
      <td>1000.00000</td>
      <td>1000.000000</td>
      <td>1000.000000</td>
      <td>1000.00000</td>
      <td>1000.000000</td>
      <td>1000.000000</td>
      <td>1000.000000</td>
      <td>1000.000000</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>203.954000</td>
      <td>38.948000</td>
      <td>546238.648000</td>
      <td>1136.000000</td>
      <td>1256.406150</td>
      <td>1.101000e+06</td>
      <td>501214.488000</td>
      <td>25126.100000</td>
      <td>-26793.700000</td>
      <td>11.644000</td>
      <td>1.83900</td>
      <td>0.992000</td>
      <td>1.487000</td>
      <td>52761.94000</td>
      <td>7433.420000</td>
      <td>7399.570000</td>
      <td>37928.950000</td>
      <td>2005.103000</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>std</th>
      <td>115.113174</td>
      <td>9.140287</td>
      <td>257063.005276</td>
      <td>611.864673</td>
      <td>244.167395</td>
      <td>2.297407e+06</td>
      <td>71701.610941</td>
      <td>27872.187708</td>
      <td>28104.096686</td>
      <td>6.951373</td>
      <td>1.01888</td>
      <td>0.820127</td>
      <td>1.111335</td>
      <td>26401.53319</td>
      <td>4880.951853</td>
      <td>4824.726179</td>
      <td>18886.252893</td>
      <td>6.015861</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>min</th>
      <td>0.000000</td>
      <td>19.000000</td>
      <td>100804.000000</td>
      <td>500.000000</td>
      <td>433.330000</td>
      <td>-1.000000e+06</td>
      <td>430104.000000</td>
      <td>0.000000</td>
      <td>-111100.000000</td>
      <td>0.000000</td>
      <td>1.00000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>100.00000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>70.000000</td>
      <td>1995.000000</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>115.750000</td>
      <td>32.000000</td>
      <td>335980.250000</td>
      <td>500.000000</td>
      <td>1089.607500</td>
      <td>0.000000e+00</td>
      <td>448404.500000</td>
      <td>0.000000</td>
      <td>-51500.000000</td>
      <td>6.000000</td>
      <td>1.00000</td>
      <td>0.000000</td>
      <td>1.000000</td>
      <td>41812.50000</td>
      <td>4295.000000</td>
      <td>4445.000000</td>
      <td>30292.500000</td>
      <td>2000.000000</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>199.500000</td>
      <td>38.000000</td>
      <td>533135.000000</td>
      <td>1000.000000</td>
      <td>1257.200000</td>
      <td>0.000000e+00</td>
      <td>466445.500000</td>
      <td>0.000000</td>
      <td>-23250.000000</td>
      <td>12.000000</td>
      <td>1.00000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>58055.00000</td>
      <td>6775.000000</td>
      <td>6750.000000</td>
      <td>42100.000000</td>
      <td>2005.000000</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>276.250000</td>
      <td>44.000000</td>
      <td>759099.750000</td>
      <td>2000.000000</td>
      <td>1415.695000</td>
      <td>0.000000e+00</td>
      <td>603251.000000</td>
      <td>51025.000000</td>
      <td>0.000000</td>
      <td>17.000000</td>
      <td>3.00000</td>
      <td>2.000000</td>
      <td>2.000000</td>
      <td>70592.50000</td>
      <td>11305.000000</td>
      <td>10885.000000</td>
      <td>50822.500000</td>
      <td>2010.000000</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>max</th>
      <td>479.000000</td>
      <td>64.000000</td>
      <td>999435.000000</td>
      <td>2000.000000</td>
      <td>2047.590000</td>
      <td>1.000000e+07</td>
      <td>620962.000000</td>
      <td>100500.000000</td>
      <td>0.000000</td>
      <td>23.000000</td>
      <td>4.00000</td>
      <td>2.000000</td>
      <td>3.000000</td>
      <td>114920.00000</td>
      <td>21450.000000</td>
      <td>23670.000000</td>
      <td>79560.000000</td>
      <td>2015.000000</td>
      <td>NaN</td>
    </tr>
  </tbody>
</table>
</div>




```python
#Unique values in each column
df.nunique()
```




    months_as_customer              391
    age                              46
    policy_number                  1000
    policy_bind_date                951
    policy_state                      3
    policy_csl                        3
    policy_deductable                 3
    policy_annual_premium           991
    umbrella_limit                   11
    insured_zip                     995
    insured_sex                       2
    insured_education_level           7
    insured_occupation               14
    insured_hobbies                  20
    insured_relationship              6
    capital-gains                   338
    capital-loss                    354
    incident_date                    60
    incident_type                     4
    collision_type                    4
    incident_severity                 4
    authorities_contacted             5
    incident_state                    7
    incident_city                     7
    incident_location              1000
    incident_hour_of_the_day         24
    number_of_vehicles_involved       4
    property_damage                   3
    bodily_injuries                   3
    witnesses                         4
    police_report_available           3
    total_claim_amount              763
    injury_claim                    638
    property_claim                  626
    vehicle_claim                   726
    auto_make                        14
    auto_model                       39
    auto_year                        21
    fraud_reported                    2
    _c39                              0
    dtype: int64




```python
#Drop unnecesary columns or columns with high cardinality, drop time colums as well for simplicity purposes for now
df = df.drop(columns=[
    'policy_number',#Unique id, no predictive value
    'insured_zip', #Identifier, Almost all unique
    'incident_location', #All Unique
    '_c39' #these are all null
])
```


```python
#convert date columns to datetime format
df['policy_bind_date'] = pd.to_datetime(df['policy_bind_date'], errors='coerce')
df['incident_date'] = pd.to_datetime(df['incident_date'], errors='coerce')
```


```python
#Extract month and years from these dates, then drop original date columns
df['policy_bind_year'] = df['policy_bind_date'].dt.year
df['policy_bind_month'] = df['policy_bind_date'].dt.month
df['incident_year'] = df['incident_date'].dt.year
df['incident_month'] = df['incident_date'].dt.month

#Drop original datetime columns
df.drop(columns=['policy_bind_date', 'incident_date'], inplace=True)
```


```python
#Check for missing values one more time
df.isnull().sum()

```




    months_as_customer             0
    age                            0
    policy_state                   0
    policy_csl                     0
    policy_deductable              0
    policy_annual_premium          0
    umbrella_limit                 0
    insured_sex                    0
    insured_education_level        0
    insured_occupation             0
    insured_hobbies                0
    insured_relationship           0
    capital-gains                  0
    capital-loss                   0
    incident_type                  0
    collision_type                 0
    incident_severity              0
    authorities_contacted          0
    incident_state                 0
    incident_city                  0
    incident_hour_of_the_day       0
    number_of_vehicles_involved    0
    property_damage                0
    bodily_injuries                0
    witnesses                      0
    police_report_available        0
    total_claim_amount             0
    injury_claim                   0
    property_claim                 0
    vehicle_claim                  0
    auto_make                      0
    auto_model                     0
    auto_year                      0
    fraud_reported                 0
    policy_bind_year               0
    policy_bind_month              0
    incident_year                  0
    incident_month                 0
    dtype: int64




```python
#Rename Fraud column, convert to binary
df['fraud'] = df['fraud_reported'].map({'Y': 1,'N': 0,}) 
df.drop(columns=['fraud_reported'], inplace=True)
```


```python
#Confirm change
print(df['fraud'].value_counts())
```

    0    753
    1    247
    Name: fraud, dtype: int64
    


```python
#Identify categorical columns and print them
categorical_cols = df.select_dtypes(include=['object']).columns.tolist()

```


```python
#print the categorical columns
print(categorical_cols)
```

    ['policy_state', 'policy_csl', 'insured_sex', 'insured_education_level', 'insured_occupation', 'insured_hobbies', 'insured_relationship', 'incident_type', 'collision_type', 'incident_severity', 'authorities_contacted', 'incident_state', 'incident_city', 'property_damage', 'police_report_available', 'auto_make', 'auto_model']
    


```python
#Take a look at number of unique values within these categorical columns
for col in categorical_cols:
    print(f"{col}: {df[col].nunique()} unique values")
```

    policy_state: 3 unique values
    policy_csl: 3 unique values
    insured_sex: 2 unique values
    insured_education_level: 7 unique values
    insured_occupation: 14 unique values
    insured_hobbies: 20 unique values
    insured_relationship: 6 unique values
    incident_type: 4 unique values
    collision_type: 4 unique values
    incident_severity: 4 unique values
    authorities_contacted: 5 unique values
    incident_state: 7 unique values
    incident_city: 7 unique values
    property_damage: 3 unique values
    police_report_available: 3 unique values
    auto_make: 14 unique values
    auto_model: 39 unique values
    


```python
#One-hot encode the remaining categorical columns
df_encoded = pd.get_dummies(df, columns=categorical_cols, drop_first=True)
df_encoded.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>months_as_customer</th>
      <th>age</th>
      <th>policy_deductable</th>
      <th>policy_annual_premium</th>
      <th>umbrella_limit</th>
      <th>capital-gains</th>
      <th>capital-loss</th>
      <th>incident_hour_of_the_day</th>
      <th>number_of_vehicles_involved</th>
      <th>bodily_injuries</th>
      <th>...</th>
      <th>auto_model_Pathfinder</th>
      <th>auto_model_RAM</th>
      <th>auto_model_RSX</th>
      <th>auto_model_Silverado</th>
      <th>auto_model_TL</th>
      <th>auto_model_Tahoe</th>
      <th>auto_model_Ultima</th>
      <th>auto_model_Wrangler</th>
      <th>auto_model_X5</th>
      <th>auto_model_X6</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>328</td>
      <td>48</td>
      <td>1000</td>
      <td>1406.91</td>
      <td>0</td>
      <td>53300</td>
      <td>0</td>
      <td>5</td>
      <td>1</td>
      <td>1</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>228</td>
      <td>42</td>
      <td>2000</td>
      <td>1197.22</td>
      <td>5000000</td>
      <td>0</td>
      <td>0</td>
      <td>8</td>
      <td>1</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>134</td>
      <td>29</td>
      <td>2000</td>
      <td>1413.14</td>
      <td>5000000</td>
      <td>35100</td>
      <td>0</td>
      <td>7</td>
      <td>3</td>
      <td>2</td>
      <td>...</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>256</td>
      <td>41</td>
      <td>2000</td>
      <td>1415.74</td>
      <td>6000000</td>
      <td>48900</td>
      <td>-62400</td>
      <td>5</td>
      <td>1</td>
      <td>1</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>228</td>
      <td>44</td>
      <td>1000</td>
      <td>1583.91</td>
      <td>6000000</td>
      <td>66000</td>
      <td>-46000</td>
      <td>20</td>
      <td>1</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 149 columns</p>
</div>




```python
from sklearn.preprocessing import StandardScaler
```


```python
#Identify numeric columns
numeric_cols = df_encoded.select_dtypes(include=['int64','float64']).columns.tolist()
numeric_cols.remove('fraud')  #keep target separate
```


```python
scaler = StandardScaler() #initialize Scaler
```


```python
#Fit and transform numeric features
df_encoded[numeric_cols] = scaler.fit_transform(df_encoded[numeric_cols])
```


```python
#Check summary stats to confirm scaling worked
print(df_encoded[numeric_cols].describe().round(2)) #Should have mean of 0 and st dev of 1
```

           months_as_customer      age  policy_deductable  policy_annual_premium  \
    count             1000.00  1000.00            1000.00                1000.00   
    mean                -0.00    -0.00              -0.00                   0.00   
    std                  1.00     1.00               1.00                   1.00   
    min                 -1.77    -2.18              -1.04                  -3.37   
    25%                 -0.77    -0.76              -1.04                  -0.68   
    50%                 -0.04    -0.10              -0.22                   0.00   
    75%                  0.63     0.55               1.41                   0.65   
    max                  2.39     2.74               1.41                   3.24   
    
           umbrella_limit  capital-gains  capital-loss  incident_hour_of_the_day  \
    count         1000.00        1000.00       1000.00                   1000.00   
    mean             0.00           0.00          0.00                     -0.00   
    std              1.00           1.00          1.00                      1.00   
    min             -0.91          -0.90         -3.00                     -1.68   
    25%             -0.48          -0.90         -0.88                     -0.81   
    50%             -0.48          -0.90          0.13                      0.05   
    75%             -0.48           0.93          0.95                      0.77   
    max              3.88           2.71          0.95                      1.63   
    
           number_of_vehicles_involved  bodily_injuries  witnesses  \
    count                      1000.00          1000.00    1000.00   
    mean                         -0.00             0.00      -0.00   
    std                           1.00             1.00       1.00   
    min                          -0.82            -1.21      -1.34   
    25%                          -0.82            -1.21      -0.44   
    50%                          -0.82             0.01      -0.44   
    75%                           1.14             1.23       0.46   
    max                           2.12             1.23       1.36   
    
           total_claim_amount  injury_claim  property_claim  vehicle_claim  \
    count             1000.00       1000.00         1000.00        1000.00   
    mean                -0.00         -0.00            0.00           0.00   
    std                  1.00          1.00            1.00           1.00   
    min                 -2.00         -1.52           -1.53          -2.01   
    25%                 -0.41         -0.64           -0.61          -0.40   
    50%                  0.20         -0.13           -0.13           0.22   
    75%                  0.68          0.79            0.72           0.68   
    max                  2.36          2.87            3.37           2.21   
    
           auto_year  policy_bind_year  policy_bind_month  incident_year  \
    count    1000.00           1000.00            1000.00         1000.0   
    mean       -0.00             -0.00              -0.00            0.0   
    std         1.00              1.00               1.00            0.0   
    min        -1.68             -1.58              -1.59            0.0   
    25%        -0.85             -0.90              -1.02            0.0   
    50%        -0.02              0.05               0.13            0.0   
    75%         0.81              0.87               0.98            0.0   
    max         1.65              1.82               1.56            0.0   
    
           incident_month  
    count         1000.00  
    mean             0.00  
    std              1.00  
    min             -0.95  
    25%             -0.95  
    50%             -0.95  
    75%              0.96  
    max              2.87  
    


```python
from sklearn.model_selection import train_test_split
```


```python
#Separate target variable
x = df_encoded.drop(columns='fraud')
y = df_encoded['fraud']

#Split into training and test set, 80% train 20% test
x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.2, random_state=444, stratify=y)
```


```python
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
```


```python
#initialize and train model
lr = LogisticRegression(max_iter=1000)
lr.fit(x_train, y_train)
```




    LogisticRegression(max_iter=1000)




```python
#Make predictions, run model on test
y_pred = lr.predict(x_test)
```


```python
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

print(confusion_matrix(y_test, y_pred)) #Confusion Matrix
print(classification_report(y_test, y_pred)) #Classification report
print("Accuracy:", accuracy_score(y_test, y_pred))#Accuracy Score
```

    [[135  16]
     [ 21  28]]
                  precision    recall  f1-score   support
    
               0       0.87      0.89      0.88       151
               1       0.64      0.57      0.60        49
    
        accuracy                           0.81       200
       macro avg       0.75      0.73      0.74       200
    weighted avg       0.81      0.81      0.81       200
    
    Accuracy: 0.815
    

|                    | Predicted Not Fraud | Predicted Fraud |
|--------------------|---------------------|-----------------|
| **Actual Not Fraud** | 135 True -                 | 16 False +              |
| **Actual Fraud**     | 21  False -                | 28 True +             |



This model correctly identified 135 non-fraud cases but missed 21 of 49 fraud cases and falsely flagged 16 non-fraud cases. This could be improved


```python
print("Train Accuracy:", lr.score(x_train, y_train))
print("Test Accuracy:", lr.score(x_test, y_test))  
```

    Train Accuracy: 0.8925
    Test Accuracy: 0.815
    

Classification Report:

 
 |             |precision    |recall  |f1-score   |support|
 |-------------|--------------|--------|----------|--------|
 | 0       |0.87      |0.89      |0.88       |151|
 | 1      | 0.64      |0.57      |0.60       | 49|


- Correctly predicted fraud 64% of the time and detects 57% of actual fraud cases, much better at predicting non-fraud.


```python
from sklearn.ensemble import RandomForestClassifier
```


```python
#Initialize next model
rf = RandomForestClassifier(n_estimators=100, random_state=444, class_weight='balanced') #100 trees in the forest, 444 = random seed, balanmced to give more weight to minority cases
```


```python
#train random forest model
rf.fit(x_train, y_train)
```




    RandomForestClassifier(class_weight='balanced', random_state=444)




```python
#Use model to make predictions
y_pred_rf = rf.predict(x_test)
```


```python
#Evaluate
print(confusion_matrix(y_test, y_pred_rf))
print(classification_report(y_test, y_pred_rf))
print("Train Accuracy:", rf.score(x_train, y_train))
print("Test Accuracy:", rf.score(x_test, y_test))
```

    [[149   2]
     [ 42   7]]
                  precision    recall  f1-score   support
    
               0       0.78      0.99      0.87       151
               1       0.78      0.14      0.24        49
    
        accuracy                           0.78       200
       macro avg       0.78      0.56      0.56       200
    weighted avg       0.78      0.78      0.72       200
    
    Train Accuracy: 1.0
    Test Accuracy: 0.78
    

- Train Accuracy of 100% is a red flag, indicative of overfitting
- 2 False + is not too bad, 42 False - is horrible, model failed to detect most actual fraud cases
- Test Accuracy is worse than before
- Logistic Regression model would be better in this case, but that one could still be improved as well.


```python

```
