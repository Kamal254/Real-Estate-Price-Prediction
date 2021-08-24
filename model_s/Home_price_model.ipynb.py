import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn import tree
from sklearn.model_selection import ShuffleSplit
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import Lasso
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import GridSearchCV
'''txt_data = pd.read_csv('bengaluru_house_prices.csv.txt')
df = pd.DataFrame(txt_data)
df.to_csv('bengaluru_i.csv', index=False)'''
data = pd.read_csv('bengaluru_i.csv')

#lets drop some unwanted columns
data_d = data.drop(['area_type', 'society', 'balcony', 'availability'], axis='columns')

#now lets drop rows only if null value rows is very small as compare to no of all rows
#is no or null value rows is big we should full by using fillna()
data1 = data_d.dropna()
data2 = data1.copy()

#now we know in 1 column like in size we have bedrooms in some bhk and diff diff
#let's have a look --- print(data2['size'].unique())

#lets create a new column to ignore unique names of bhk
#x:int(x.split(' ')[0] will change 3 BHK or 2 Bedrooms in to only no 3,2

data2['BHK'] = data2['size'].apply(lambda  x:int(x.split(' ')[0]))

#similarly we need to explore our left data features like size bath location etc
# while we print print(data2.total_sqft.unique()) we get that some values are in range ex = 3290-3450 now lets take average of this and modify our data
'''def is_float(x):
    try:
        float(x)
    except:
        return False
    return True

print(data2[~data2['total_sqft'].apply(is_float)])'''
# we can see that 190 lines are there which dont have a no
#  print(data2['total_sqft'].apply(is_float)) basically this will tell us true or false to check tha values and print them for false


#Below function will set range values into average and astrign values into None
def convert_range_to_num(x):
    parts = x.split('-')
    if len(parts)==2:
        new_val = (float(parts[0])+float(parts[1]))//2
        return new_val
    try:
        return float(x)
    except:
        return None

#whenever we doing change in our orignal data we should do a deep copy and create a new dataframe
data3 = data2.copy()
data3['total_sqft'] = data3['total_sqft'].apply(convert_range_to_num)
#as above we know we put some values None lets count them and delete is no if count is low
data4 = data3.dropna()

#now create some our own predicting features using given data
#example = price_per_sqft
data5 = data4.copy()
data5['price_per_sqft'] = data5['price']*100000/data5['total_sqft']


#now as we know we have diff diff location as well let's calculate
'''print(len(data5['location'].unique()))'''
#we got 1298 which is too big no to do Encoding we need less no to make our data a bit small
#we are not going to add 1298 more dummies column so...
#let's look at the statestics of our location column

#data5.location = data5.location.apply(lambda x:x.strip())
location_stats = data5.groupby('location')['location'].agg('count').sort_values(ascending=False)
'''print(location_stats)'''
#after looking to states we can see that too many locations have only single house lets merge them

#lets calculate how many location having less than 10 datasets
location_stats_less_than_10 = location_stats[location_stats<=10]
'''print(location_stats_less_than_10)#we got 1058 location'''

#below we apply a condition to merge the location
data5['location'] = data5['location'].apply(lambda x:'other' if x in location_stats_less_than_10 else x)
#now we have 241 diff location which is very less than 1298
'''print(len(data5.location.unique()))'''

#remove outlier outliers are data errors as we know in 1000sqft no 6 bedrooms can fit
'''print(data5[data5.total_sqft/data5.BHK<300])'''

#here we removed outliers
data6 = data5[~(data5.total_sqft/data5.BHK<300)]

#removing extream cases is also good for our data
'''print(data6.price_per_sqft.describe())''' #by print this we will get statstics like max min etc

def remove_pps_outliers(data6):
    df_out = pd.DataFrame()
    for key, subdf in data6.groupby('location'):
        mean = np.mean(subdf.price_per_sqft)
        sd = np.std(subdf.price_per_sqft)
        reduced_data6 = subdf[(subdf.price_per_sqft>(mean-sd)) & (subdf.price_per_sqft<=(mean+sd))]
        df_out = pd.concat([df_out, reduced_data6], ignore_index=True)
    return df_out
data7 = remove_pps_outliers(data6)


def plot_scater_chart(df, location):
    bhk2 = df[(df.location == location) & (df.BHK == 2)]
    bhk3 = df[(df.location == location) & (df.BHK == 3)]
    plt.rcParams['figure.figsize'] = (15, 10)
    plt.scatter(bhk2.total_sqft, bhk2.price, color='blue', label='2 BHK', s=50)
    plt.scatter(bhk3.total_sqft, bhk3.price, marker='+', color='green', label='3 BHK', s=50)
    plt.xlabel("Total Square Feet Area")
    plt.ylabel("Price (Lakh Indian Rupees)")
    plt.title(location)
    plt.legend()
    #plt.show()
#plot_scater_chart(data7, "Hebbal")

#abive is the graph visualisation
#Now looke that we have some 3bhk flat whose price is equal and lower than 2bhk flat
#in same location so we need to remove them as well

def remove_bhk_outliers(df):
    exclude_indices = np.array([])
    for location, location_df in df.groupby('location'):
        bhk_stats = {}
        for bhk, bhk_df in location_df.groupby('BHK'):
            bhk_stats[bhk] = {
                'mean':np.mean(bhk_df.price_per_sqft),
                'std':np.std(bhk_df.price_per_sqft),
                'count':bhk_df.shape[0]
            }
        for bhk, bhk_df in location_df.groupby("BHK"):
            states = bhk_stats.get(bhk-1)
            if states and states['count']>5:
                exclude_indices = np.append(exclude_indices, bhk_df[bhk_df.price_per_sqft<(states['mean'])].index.values)
    return df.drop(exclude_indices, axis='index')
data8 = remove_bhk_outliers(data7)
plot_scater_chart(data8, "Hebbal")

data9 = data8[data8.bath<data8.BHK+2]
#final step lets drop some columns which are not useful as feature

data10 = data9.drop(['size','price_per_sqft'], axis='columns')



'''Model'''
#step 1---> one hot encoding for location
location_dummies = pd.get_dummies(data10.location)
data11 = pd.concat([data10, location_dummies.drop('other', axis = 'columns')], axis = 'columns')#we droped last column bcs of trap error
data12 = data11.drop('location', axis = 'columns')

#step 2---> we need x and y for model
X = data12.drop('price', axis='columns')
Y = data12.price
x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=10)

#step 3---> let's create model
#try 1st model is linear regression
lr_model = LinearRegression()
lr_model.fit(X, Y)
#print(lr_model.score(x_test, y_test))

#But as we know we can use k_fold cross velidation to select best model to fit our data
cv = ShuffleSplit(n_splits=5, test_size=0.2, random_state=0)
lr_score = cross_val_score(LinearRegression(), X, Y, cv=cv)#thsi will return me the array of 5 score

#Grid_search_cv will let us know which model to select
def find_best_model(X,Y):
    algo = {
        'linear_regression':{
            'model':LinearRegression(),
            'params':{
                'normalize':[True, False]
            }
        },
        'lasso':{
            'model':Lasso(),
            'params':{
                'alpha':[1,2],
                'selection':['random', 'cyclic']
            }
        },
        'dicision_tree':{
            'model':DecisionTreeRegressor(),
            'params':{
                'criterion':['mse', 'friedman_mse'],
                'splitter':['best', 'random']
            }
        }
    }
    score = []
    cv = ShuffleSplit(n_splits=5, test_size=0.2, random_state=0)
    for model , config in algo.items():
        gs = GridSearchCV(config['model'], config['params'], cv = cv, return_train_score=False)
        gs.fit(X,Y)
        score.append({
            'model':model,
            'best_score':gs.best_score_,
            'best_params':gs.best_params_
        })
    return pd.DataFrame(score, columns=['model', 'best_score', 'best_params'])

#print(find_best_model(X,Y))# as after print this we can see that linear_regressoin is best model

def predict_price(location, sqft, bath, BHK):
    loc_index = np.where(X.columns == location )[0][0]
    x = np.zeros(len(X.columns))#this is an array of zeros of size no of columns
    x[0] = sqft
    x[1]= bath
    x[2] = BHK
    if loc_index>=0:
        x[loc_index] = 1
    #print(x)
    return lr_model.predict([x])[0]
print(predict_price('1st Phase JP Nagar', 1000, 2, 3))


import pickle
import json
with open('model1.pickle', 'wb') as f:
    pickle.dump(lr_model,f)


columns = {
    'data_columns' : [col.lower() for col in X.columns]
}
with open("columns1.json","w") as f:
    f.write(json.dumps(columns))