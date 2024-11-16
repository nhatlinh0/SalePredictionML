import pandas as pd
import numpy as np
import tkinter as tk
from tkinter import filedialog, messagebox
from matplotlib import pyplot as plt
from matplotlib.gridspec import GridSpec
import seaborn as sns
from scipy import stats
from scipy.special import boxcox1p
from sklearn.model_selection import train_test_split
from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import KFold
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from sklearn.feature_selection import RFE
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import LabelEncoder
import time
import warnings
import pickle # for saving the model
from pandasql import sqldf

warnings.filterwarnings("ignore") # ignoring annoying warnings
pysqldf = lambda q: sqldf(q, globals())
data_path = 'D:/ML/Sales-Forecasting/kaggle/inputs/dataset/'

# Load datasets
features = pd.read_csv(f'{data_path}features.csv')
train = pd.read_csv(f'{data_path}train.csv')
stores = pd.read_csv(f'{data_path}stores.csv')
test = pd.read_csv(f'{data_path}test.csv')
sample_submission = pd.read_csv(f'{data_path}sampleSubmission.csv')

feat_sto = features.merge(stores, how='inner', on='Store')
feat_sto.Date = pd.to_datetime(feat_sto.Date)
train.Date = pd.to_datetime(train.Date)
test.Date = pd.to_datetime(test.Date)
feat_sto['Week'] = feat_sto.Date.dt.isocalendar().week
feat_sto['Year'] = feat_sto.Date.dt.year
train_detail = train.merge(feat_sto, 
                           how='inner',
                           on=['Store','Date','IsHoliday']).sort_values(by=['Store','Dept','Date']).reset_index(drop=True)
test_detail = test.merge(feat_sto, 
                           how='inner',
                           on=['Store','Date','IsHoliday']).sort_values(by=['Store',
                                                                            'Dept',
                                                                            'Date']).reset_index(drop=True)
del features, train, stores, test
null_columns = (train_detail.isnull().sum(axis = 0)/len(train_detail)).sort_values(ascending=False).index
null_data = pd.concat([
    train_detail.isnull().sum(axis = 0),
    (train_detail.isnull().sum(axis = 0)/len(train_detail)).sort_values(ascending=False),
    train_detail.loc[:, train_detail.columns.isin(list(null_columns))].dtypes], axis=1)
null_data = null_data.rename(columns={0: '# null', 
                                      1: '% null', 
                                      2: 'type'}).sort_values(ascending=False, by = '% null')
null_data = null_data[null_data["# null"]!=0]
null_data

pysqldf("""
SELECT
    T.*,
    case
        when ROW_NUMBER() OVER(partition by Year order by week) = 1 then 'Super Bowl'
        when ROW_NUMBER() OVER(partition by Year order by week) = 2 then 'Labor Day'
        when ROW_NUMBER() OVER(partition by Year order by week) = 3 then 'Thanksgiving'
        when ROW_NUMBER() OVER(partition by Year order by week) = 4 then 'Christmas'
    end as Holyday,
    case
        when ROW_NUMBER() OVER(partition by Year order by week) = 1 then 'Sunday'
        when ROW_NUMBER() OVER(partition by Year order by week) = 2 then 'Monday'
        when ROW_NUMBER() OVER(partition by Year order by week) = 3 then 'Thursday'
        when ROW_NUMBER() OVER(partition by Year order by week) = 4 and Year = 2010 then 'Saturday'
        when ROW_NUMBER() OVER(partition by Year order by week) = 4 and Year = 2011 then 'Sunday'
        when ROW_NUMBER() OVER(partition by Year order by week) = 4 and Year = 2012 then 'Tuesday'
    end as Day
    from(
        SELECT DISTINCT
            Year,
            Week,
            case 
                when Date <= '2012-11-01' then 'Train Data' else 'Test Data' 
            end as Data_type
        FROM feat_sto
        WHERE IsHoliday = True) as T""")


# weekly_sales_2010 = train_detail[train_detail.Year==2010]['Weekly_Sales'].groupby(train_detail['Week']).mean()
# weekly_sales_2011 = train_detail[train_detail.Year==2011]['Weekly_Sales'].groupby(train_detail['Week']).mean()
# weekly_sales_2012 = train_detail[train_detail.Year==2012]['Weekly_Sales'].groupby(train_detail['Week']).mean()
# plt.figure(figsize=(20,10))
# sns.lineplot(x=weekly_sales_2010.index, y=weekly_sales_2010.values)
# sns.lineplot(x=weekly_sales_2011.index, y=weekly_sales_2011.values)
# sns.lineplot(x=weekly_sales_2012.index, y=weekly_sales_2012.values)
# plt.grid()
# plt.xticks(np.arange(1, 53, step=1))
# plt.legend(['2010', '2011', '2012'], loc='best', fontsize=16)
# plt.title('Average Weekly Sales - Per Year', fontsize=18)
# plt.ylabel('Sales', fontsize=16)
# plt.xlabel('Week', fontsize=16)
# plt.show()

# train_detail.loc[(train_detail.Year==2010) & (train_detail.Week==13), 'IsHoliday'] = True
# train_detail.loc[(train_detail.Year==2011) & (train_detail.Week==16), 'IsHoliday'] = True
# train_detail.loc[(train_detail.Year==2012) & (train_detail.Week==14), 'IsHoliday'] = True
# test_detail.loc[(test_detail.Year==2013) & (test_detail.Week==13), 'IsHoliday'] = True

# weekly_sales_mean = train_detail['Weekly_Sales'].groupby(train_detail['Date']).mean()
# weekly_sales_median = train_detail['Weekly_Sales'].groupby(train_detail['Date']).median()

# plt.figure(figsize=(20,8))
# sns.lineplot(x=weekly_sales_mean.index, y=weekly_sales_mean.values)
# sns.lineplot(x=weekly_sales_median.index, y=weekly_sales_median.values)

# plt.grid()
# plt.legend(['Mean', 'Median'], loc='best', fontsize=16)
# plt.title('Weekly Sales - Mean and Median', fontsize=18)
# plt.ylabel('Sales', fontsize=16)
# plt.xlabel('Date', fontsize=16)
# plt.show()

# weekly_sales = train_detail['Weekly_Sales'].groupby(train_detail['Store']).mean()
# plt.figure(figsize=(20,10))
# sns.barplot(x=weekly_sales.index, y=weekly_sales.values, palette='dark')
# plt.grid()
# plt.title('Average Sales - per Store', fontsize=18)
# plt.ylabel('Sales', fontsize=16)
# plt.xlabel('Store', fontsize=16)
# plt.show()

# weekly_sales = train_detail['Weekly_Sales'].groupby(train_detail['Dept']).mean()
# plt.figure(figsize=(20,10))
# sns.barplot(x=weekly_sales.index, y=weekly_sales.values, palette='dark')
# plt.grid()
# plt.title('Average Sales - per Dept', fontsize=18)
# plt.ylabel('Sales', fontsize=16)
# plt.xlabel('Dept', fontsize=16)
# plt.show()

# train_detail_numeric = train_detail.apply(pd.to_numeric, errors='coerce')
# train_detail_numeric = train_detail_numeric.dropna(axis=1, how='all')
# sns.set(style="white")
# corr = train_detail_numeric.corr()
# mask = np.triu(np.ones_like(corr, dtype=np.bool))
# f, ax = plt.subplots(figsize=(20, 20))
# cmap = sns.diverging_palette(220, 10, as_cmap=True)
# plt.title('Correlation Matrix', fontsize=18)
# sns.heatmap(corr, mask=mask, cmap=cmap, vmax=.3, center=0, square=True, linewidths=.5, cbar_kws={"shrink": .5}, annot=True)
# plt.show()

train_detail = train_detail.drop(columns=['Fuel_Price','MarkDown1','MarkDown2','MarkDown3','MarkDown4','MarkDown5'])
test_detail = test_detail.drop(columns=['Fuel_Price','MarkDown1','MarkDown2','MarkDown3','MarkDown4','MarkDown5'])

# def boxplot(feature):
#     fig = plt.figure(figsize=(20,8))
#     gs = GridSpec(1,2)
#     sns.boxplot(y=train_detail.Weekly_Sales, x=train_detail[feature], ax=fig.add_subplot(gs[0,0]))
#     plt.ylabel('Sales', fontsize=16)
#     plt.xlabel(feature, fontsize=16)
#     sns.stripplot(y=train_detail.Weekly_Sales, x=train_detail[feature], ax=fig.add_subplot(gs[0,1]))
#     plt.ylabel('Sales', fontsize=16)
#     plt.xlabel(feature, fontsize=16)
#     plt.show()

# def boxcox(feature):
#     fig = plt.figure(figsize=(18,15))
#     gs = GridSpec(2,2)
#     j = sns.scatterplot(y=train_detail['Weekly_Sales'], 
#                         x=boxcox1p(train_detail[feature], 0.15), ax=fig.add_subplot(gs[0,1]), palette = 'blue')

#     plt.title('BoxCox 0.15\n' + 'Corr: ' + str(np.round(train_detail['Weekly_Sales'].corr(boxcox1p(train_detail[feature], 0.15)),2)) +
#               ', Skew: ' + str(np.round(stats.skew(boxcox1p(train_detail[feature], 0.15), nan_policy='omit'),2)))
    
#     j = sns.scatterplot(y=train_detail['Weekly_Sales'], 
#                         x=boxcox1p(train_detail[feature], 0.25), ax=fig.add_subplot(gs[1,0]), palette = 'blue')

#     plt.title('BoxCox 0.25\n' + 'Corr: ' + str(np.round(train_detail['Weekly_Sales'].corr(boxcox1p(train_detail[feature], 0.25)),2)) +
#               ', Skew: ' + str(np.round(stats.skew(boxcox1p(train_detail[feature], 0.25), nan_policy='omit'),2)))
    
#     j = sns.distplot(train_detail[feature], ax=fig.add_subplot(gs[1,1]), color = 'green')

#     plt.title('Distribution\n')
    
#     j = sns.scatterplot(y=train_detail['Weekly_Sales'], 
#                         x=train_detail[feature], ax=fig.add_subplot(gs[0,0]), color = 'red')

#     plt.title('Linear\n' + 'Corr: ' + str(np.round(train_detail['Weekly_Sales'].corr(train_detail[feature]),2)) + ', Skew: ' + 
#                str(np.round(stats.skew(train_detail[feature], nan_policy='omit'),2)))
    
#     plt.show()

train_detail.Type = train_detail.Type.apply(lambda x: 3 if x == 'A' else(2 if x == 'B' else 1))
test_detail.Type = test_detail.Type.apply(lambda x: 3 if x == 'A' else(2 if x == 'B' else 1))
# boxplot('IsHoliday')
# boxplot('Type')
# boxcox('Temperature')

train_detail = train_detail.drop(columns=['Temperature'])
test_detail = test_detail.drop(columns=['Temperature'])
# boxcox('Unemployment')
train_detail = train_detail.drop(columns=['Unemployment'])
test_detail = test_detail.drop(columns=['Unemployment'])
# boxcox('CPI')
train_detail = train_detail.drop(columns=['CPI'])
test_detail = test_detail.drop(columns=['CPI'])
# boxcox('Size')

def WMAE(dataset, real, predicted):
    weights = dataset.IsHoliday.apply(lambda x: 5 if x else 1)
    return np.round(np.sum(weights*abs(real-predicted))/(np.sum(weights)), 2)

def MAE(real, predicted):
    return np.mean(np.abs(real - predicted))

def MSE(real, predicted):
    return np.mean((real - predicted) ** 2)

def RMSE(real, predicted):
    return np.sqrt(MSE(real, predicted))

def R2(real, predicted):
    return r2_score(real, predicted)

def knn():
    knn = KNeighborsRegressor(n_neighbors=10)
    return knn

def extraTreesRegressor():
    clf = ExtraTreesRegressor(n_estimators=60, max_features=3, verbose=1, n_jobs=1)
    return clf
    
def randomForestRegressor():
    clf = RandomForestRegressor(n_estimators = 100, max_features = 'log2', verbose = 1, bootstrap = True)
    return clf

def linear_reg():
    regr = LinearRegression()
    return regr

def predict_(m, test_x):
    return pd.Series(m.predict(test_x))

def model_():
    # return knn()
#     return extraTreesRegressor()
    return randomForestRegressor()
    # return linear_reg()

def train_(train_x, train_y):
    m = model_()
    m.fit(train_x, train_y)
    return m

def train_and_predict(train_x, train_y, test_x):
    m = train_(train_x, train_y)
    return predict_(m, test_x), m

# kf = KFold(n_splits=5)
# splited = []
# # dataset2 = dataset.copy()
# for name, group in train_detail.groupby(["Store", "Dept"]):
#     group = group.reset_index(drop=True)
#     trains_x = []
#     trains_y = []
#     tests_x = []
#     tests_y = []
#     if group.shape[0] <= 5:
#         f = np.array(range(5))
#         np.random.shuffle(f)
#         group['fold'] = f[:group.shape[0]]
#         continue
#     fold = 0
#     for train_index, test_index in kf.split(group):
#         group.loc[test_index, 'fold'] = fold
#         fold += 1
#     splited.append(group)

# splited = pd.concat(splited).reset_index(drop=True)

# best_model = None
# error_cv = 0
# best_error = np.iinfo(np.int32).max
# mae_cv = 0
# mse_cv = 0
# rmse_cv = 0
# r2_cv = 0
# for fold in range(5):
#     train_detail2 = splited.loc[splited['fold'] != fold]
#     test_detail2 = splited.loc[splited['fold'] == fold]
#     train_y = train_detail2['Weekly_Sales']
#     train_x = train_detail2[['Store','Dept','IsHoliday','Size','Week','Type','Year']]
#     test_y = test_detail2['Weekly_Sales']
#     test_x = test_detail2[['Store', 'Dept', 'IsHoliday', 'Size', 'Week', 'Type', 'Year']]
#     print(train_detail2.shape, test_detail2.shape)

#     predicted, model = train_and_predict(train_x, train_y, test_x)

#     error = WMAE(test_x, test_y, predicted)
#     mae = MAE(test_y, predicted)
#     mse = MSE(test_y, predicted)
#     rmse = RMSE(test_y, predicted)
#     r2 = R2(test_y, predicted)
    
#     print(error)
#     error_cv += error
#     mae_cv += mae
#     mse_cv += mse
#     rmse_cv += rmse
#     r2_cv += r2
    
#     print(f'Mean MAE: {mae_cv}')
#     print(f'Mean MSE: {mse_cv}')
#     print(f'Mean RMSE: {rmse_cv}')
#     print(f'Mean R²: {r2_cv}')
#     print(fold, error)
#     if error < best_error:
#         print('Find best model')
#         best_error = error
#         best_model = model

# error_cv /= 5
# mae_cv /= 5
# mse_cv /= 5
# rmse_cv /= 5
# r2_cv /= 5
# print(f'Mean MAE: {mae_cv}')
# print(f'Mean MSE: {mse_cv}')
# print(f'Mean RMSE: {rmse_cv}')
# print(f'Mean R²: {r2_cv}')
# print(fold, error_cv)


def random_forest(n_estimators, max_depth):
    result = []
    for estimator in n_estimators:
        for depth in max_depth:
            wmaes_cv = []
            for i in range(1,5):
                print('k:', i, ', n_estimators:', estimator, ', max_depth:', depth)
                x_train, x_test, y_train, y_test = train_test_split(X_train, Y_train, test_size=0.3)
                RF = RandomForestRegressor(n_estimators=estimator, max_depth=depth)
                RF.fit(x_train, y_train)
                predicted = RF.predict(x_test)
                wmaes_cv.append(WMAE(x_test, y_test, predicted))
            print('WMAE:', np.mean(wmaes_cv))
            result.append({'Max_Depth': depth, 'Estimators': estimator, 'WMAE': np.mean(wmaes_cv)})
    return pd.DataFrame(result)

def random_forest_II(n_estimators, max_depth, max_features):
    result = []
    for feature in max_features:
        wmaes_cv = []
        for i in range(1,5):
            print('k:', i, ', max_features:', feature)
            x_train, x_test, y_train, y_test = train_test_split(X_train, Y_train, test_size=0.3)
            RF = RandomForestRegressor(n_estimators=n_estimators, max_depth=max_depth, max_features=feature)
            RF.fit(x_train, y_train)
            predicted = RF.predict(x_test)
            wmaes_cv.append(WMAE(x_test, y_test, predicted))
        print('WMAE:', np.mean(wmaes_cv))
        result.append({'Max_Feature': feature, 'WMAE': np.mean(wmaes_cv)})
    return pd.DataFrame(result)

def random_forest_III(n_estimators, max_depth, max_features, min_samples_split, min_samples_leaf):
    result = []
    for split in min_samples_split:
        for leaf in min_samples_leaf:
            wmaes_cv = []
            for i in range(1,5):
                print('k:', i, ', min_samples_split:', split, ', min_samples_leaf:', leaf)
                x_train, x_test, y_train, y_test = train_test_split(X_train, Y_train, test_size=0.3)
                RF = RandomForestRegressor(n_estimators=n_estimators, max_depth=max_depth, max_features=max_features, 
                                           min_samples_leaf=leaf, min_samples_split=split)
                RF.fit(x_train, y_train)
                predicted = RF.predict(x_test)
                wmaes_cv.append(WMAE(x_test, y_test, predicted))
            print('WMAE:', np.mean(wmaes_cv))
            result.append({'Min_Samples_Leaf': leaf, 'Min_Samples_Split': split, 'WMAE': np.mean(wmaes_cv)})
    return pd.DataFrame(result)

X_train = train_detail[['Store','Dept','IsHoliday','Size','Week','Type','Year']]
Y_train = train_detail['Weekly_Sales']

# n_estimators = [56, 58, 60]
# max_depth = [25, 27, 30]
# random_forest(n_estimators, max_depth)

# max_features = [2, 3, 4, 5, 6, 7]
# random_forest_II(n_estimators=56, max_depth=30, max_features=max_features)

# min_samples_split = [2, 3, 4]
# min_samples_leaf = [1, 2, 3]
# random_forest_III(n_estimators=56, max_depth=30, max_features=7, min_samples_split=min_samples_split, min_samples_leaf=min_samples_leaf)

RF = RandomForestRegressor(n_estimators=56, max_depth=30, max_features=7, min_samples_split=3, min_samples_leaf=1)
RF.fit(X_train, Y_train)

model_save_file = open("SavedModel.sav", 'wb')
pickle.dump(RF, model_save_file)
model_save_file.close()

 # load_saved_model_file = open('SavedModel.sav', 'rb')
# loaded_model = pickle.load(load_saved_model_file)
# load_saved_model_file.close()

# X_test = test_detail[['Store', 'Dept', 'IsHoliday', 'Size', 'Week', 'Type', 'Year']]
# loaded_predict = loaded_model.predict(X_test)

# X_test = test_detail[['Store', 'Dept', 'IsHoliday', 'Size', 'Week', 'Type', 'Year']]
# predict = RF.predict(X_test)
# print(np.all(loaded_predict==predict))
# sample_submission['Store', 'Dept', 'Week', 'Year','Weekly_Sales'] = loaded_predict

# sample_submission.to_csv('sampleSubmission.csv', index=False)

def preprocess_data(df):
    # Tạo từ điển ánh xạ cho cột 'Type'
    type_mapping = {'A': 3, 'B': 2, 'C': 1}
    df['Type'] = df['Type'].map(type_mapping)
    return df

def run_model():
    try:
        # Đọc dữ liệu từ file CSV
        file_path = filedialog.askopenfilename(title="Chọn tệp dữ liệu", filetypes=(("CSV files", "*.csv"), ("All files", "*.*")))
        if not file_path:
            return
        test_detail = pd.read_csv(file_path)
        test_detail['IsHoliday'] = test_detail['IsHoliday'].apply(lambda x: 1 if x == 'TRUE' else 0)
        test_detail = preprocess_data(test_detail)
        # Tiền xử lý dữ liệu đầu vào
        X_test = test_detail[['Store', 'Dept', 'IsHoliday', 'Size', 'Week', 'Type', 'Year']]

        # Load mô hình đã lưu
        with open('SavedModel.sav', 'rb') as model_file:
            loaded_model = pickle.load(model_file)

        # Dự đoán với mô hình đã tải
        loaded_predict = loaded_model.predict(X_test)
        # Lưu kết quả vào DataFrame
        sample_submission = test_detail[['Store', 'Dept', 'Week', 'Year']]
        sample_submission['Weekly_Sales'] = loaded_predict

        # Lưu kết quả vào tệp CSV
        output_file = filedialog.asksaveasfilename(defaultextension=".csv", filetypes=(("CSV files", "*.csv"), ("All files", "*.*")))
        if output_file:
            
            sample_submission.to_csv(output_file, index=False)
            messagebox.showinfo("Success", f"Đã lưu kết quả vào: {output_file}")
        else:
            messagebox.showwarning("Warning", "Bạn chưa chọn tệp để lưu.")
        
    except Exception as e:
        messagebox.showerror("Error", f"Đã xảy ra lỗi: {e}")

root = tk.Tk()
root.title("DỰ ĐOÁN DOANH SỐ BÁN HÀNG")
root.geometry("600x400")

title_label = tk.Label(root, text="DỰ ĐOÁN DOANH SỐ BÁN HÀNG", font=("Arial", 16, "bold"))
title_label.pack(pady=10)

subtitle_label = tk.Label(root, text="Chọn tệp CSV theo mẫu \n(Store, Dept, IsHoliday, Size, Week, Type, Year)", font=("Arial", 12))
subtitle_label.pack(pady=5)

subtitle_label = tk.Label(root, text="VD:1,2,1,A,5,2024", font=("Arial", 12))
subtitle_label.pack(pady=5)

run_button = tk.Button(root, text="Chạy Mô Hình", command=run_model)
run_button.pack(pady=50)

root.mainloop()