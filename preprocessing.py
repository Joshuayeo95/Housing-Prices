''' Preprcoessing Data for the Housing Price dataset.'''

### Importing Libraries ### 
# Standard Libraries #
from scipy import stats
import pandas as pd
import numpy as np
import pickle
import time

# Visualisation #
import matplotlib.pyplot as plt 
import seaborn as sns
from matplotlib.offsetbox import AnchoredText

### Defining Functions ###

def missing_pct(df, assign=False):
    '''
    Function that shows variables that have missing values and the percentage of total observations that are missing.
    Pass assign=True if user wishes to save the resulting series.
    '''
    
    percentage_missing = df.isnull().mean().sort_values(ascending=False) * 100
    percentage_missing = percentage_missing.loc[percentage_missing > 0]
    
    if len(percentage_missing) > 0:
        print('The following variables have missing data and the percentage of total missing are as follows:')
        print(percentage_missing)
    
    else:
        print('The dataframe has no missing values in any column.')
    
    if assign:
        return percentage_missing


def var_plots(series, target=True):
    '''
    Function that plots the distribution of the variable as well as its distribution against the Target LogSales=Price.
    
    :params:
        series : Feature column in the dataset.
        target : Set to true to plot variable against target variable LogSalePrice. Default is True.
    '''
    if target:
        f, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2,2,figsize=(10,8))
        f.tight_layout(pad=5.0)
    
        ax1.set_title('Distribution for Variable: {}'.format(series.name), size=12)
        _, p_val = stats.normaltest(series)
        anchored_text = AnchoredText('Skew = {:.2f}\nKurtosis = {:.2f}\nP-val = {:.4f}'.format(stats.skew(series),
                                                                                               stats.kurtosis(series),
                                                                                               p_val), loc='center right', frameon=False)
        ax1.add_artist(anchored_text)
        
        sns.distplot(series, bins=100, ax=ax1)

        stats.probplot(series, plot=ax2)
        
        sns.boxplot(series, ax=ax3)
        ax3.set_title('Boxplot for {}'.format(series.name), size=12, y=1.05)
        
        sns.scatterplot(x=series, y=df.LogSalePrice, alpha=0.5, ax=ax4)
        correlation = series.corr(df.LogSalePrice)
        ax4.set_title('Correlation = {:.2f}'.format(correlation), size=12, y=1.05)
    
    else:
        f, (ax1, ax2) = plt.subplots(1,2,figsize=(10,4))
        f.tight_layout(pad=5.0)
    
        ax1.set_title('Distribution for Variable: {}'.format(series.name), size=12)
        _, p_val = stats.normaltest(series)
        anchored_text = AnchoredText('Skew = {:.2f}\nKurtosis = {:.2f}\nP-val = {:.4f}'.format(stats.skew(series),
                                                                                               stats.kurtosis(series),
                                                                                               p_val), loc='center right', frameon=False)
        ax1.add_artist(anchored_text)
        sns.distplot(series, bins=100, ax=ax1)

        stats.probplot(series, plot=ax2)
    sns.despine();


# Function to display labels on plots
def show_values_on_bars(axs):
    def _show_on_single_plot(ax):        
        for p in ax.patches:
            _x = p.get_x() + p.get_width() / 2
            _y = p.get_y() + p.get_height()
            value = '{:.0f}'.format(p.get_height())
            ax.text(_x, _y, value, ha="center") 

    if isinstance(axs, np.ndarray):
        for idx, ax in np.ndenumerate(axs):
            _show_on_single_plot(ax)
    else:
        _show_on_single_plot(axs)


def categorical_var_plots(series, fig_size=(10,5), rotate=False, rotate_angle=30):
    '''
    Function that provides a countplot to see the distribution of the different categories for each variable.
    Also plots the catplot against LogSalePrice.
    '''
    f, (ax1, ax2) = plt.subplots(1,2,figsize=fig_size)
    f.tight_layout(pad=5.0)
    
    count_plt = sns.countplot(series, palette='pastel', ax=ax1)
    ax1.set_title('Occurences of Each Category', size=14, y=1.05)
    show_values_on_bars(ax1)
    
    box_plt = sns.boxplot(x=series.name, y='LogSalePrice', data=df, palette='pastel', ax=ax2)
    ax2.set_title('Categorical Plot of against Logged Prices', size=14, y=1.05)
    
    if rotate:
        count_plt.set_xticklabels(count_plt.get_xticklabels(), rotation=rotate_angle, fontsize=12)
        box_plt.set_xticklabels(box_plt.get_xticklabels(), rotation=rotate_angle, fontsize=12)
    
    sns.despine();



# Loading in the dataset
df = pd.read_csv('./Data/train.csv')
print('The data frame has {} rows and {} columns.'.format(df.shape[0], df.shape[1]))

### Handling Missing Values ###
missing_pct(df) # Prints out the columns that have missing data and the percentage missing from each column

print('Variables with more than 90% missing values will be dropped.')
df = df[df.columns[df.isnull().mean() < 0.9]] # Dropping variables with more than 90% missing values

df.Fence = df.Fence.fillna('NoFence')

garage_details = ['GarageType', 'GarageFinish', 'GarageQual', 'GarageCond'] # Features regarding garage
df[garage_details] = df[garage_details].fillna('NoGarage')

basement_details = ['BsmtFinType2', 'BsmtExposure', 'BsmtFinType1', 'BsmtCond', 'BsmtQual'] # Features regarding basement
df[basement_details] = df[basement_details].fillna('NoBsmt')

print('As the number of missing data for the following variables are low, we will just be dropping the observations that have missing data for these variables.')
df = df.dropna(how='any', subset=['MasVnrType', 'MasVnrArea', 'Electrical'])

print('For the variable LotFrontage, we we will be using K-Nearest Neighbours to impute the missing data.')
from missingpy import KNNImputer
imputer = KNNImputer(n_neighbors=5,
                     weights='distance',
                     metric='masked_euclidean')

df.LotFrontage = imputer.fit_transform(np.array(df.drop('FireplaceQu', axis=1).LotFrontage).reshape(-1,1))

df.FireplaceQu = df.FireplaceQu.fillna('NoFireplc')

print('Checking for any more columns with missing data ...')
missing_pct(df)


### Checking Data Types ### 
print('Changing numeric data to categorical ...')
df = df.replace({'MSSubClass' : {20 : "SC20", 30 : "SC30", 40 : "SC40", 45 : "SC45", 
                                 50 : "SC50", 60 : "SC60", 70 : "SC70", 75 : "SC75", 
                                 80 : "SC80", 85 : "SC85", 90 : "SC90", 120 : "SC120", 
                                 150 : "SC150", 160 : "SC160", 180 : "SC180", 190 : "SC190"},
                 'MoSold' : {1 : "Jan", 2 : "Feb", 3 : "Mar", 4 : "Apr", 5 : "May", 6 : "Jun",
                            7 : "Jul", 8 : "Aug", 9 : "Sep", 10 : "Oct", 11 : "Nov", 12 : "Dec"}
                })

print('Changing numeric variables to an interval scale as they are ordinal in nature ...')
df = df.replace({'ExterQual' : {'Po' : 1, 'Fa' : 2, 'TA' : 3, 'Gd' : 4, 'Ex' : 5},
                 'ExterCond' : {'Po' : 1, 'Fa' : 2, 'TA' : 3, 'Gd' : 4, 'Ex' : 5},
                 'BsmtQual' : {'NoBsmt' : 0, 'Po' : 1, 'Fa' : 2, 'TA' : 3, 'Gd' : 4, 'Ex' : 5},
                 'BsmtCond' : {'NoBsmt' : 0, 'Po' : 1, 'Fa' : 2, 'TA' : 3, 'Gd' : 4, 'Ex' : 5},
                 'BsmtExposure' : {'NoBsmt' : 0, 'No' : 1, 'Mn' : 2, 'Av' : 3, 'Gd' : 4},
                 'BsmtFinType1' : {'NoBsmt' : 0, 'Unf' : 1, 'LwQ' : 2, 'Rec' : 3, 'BLQ' : 4, 'ALQ' : 5, 'GLQ' : 6},
                 'BsmtFinType2' : {'NoBsmt' : 0, 'Unf' : 1, 'LwQ' : 2, 'Rec' : 3, 'BLQ' : 4, 'ALQ' : 5, 'GLQ' : 6},
                 'HeatingQC' : {'Po' : 1, 'Fa' : 2, 'TA' : 3, 'Gd' : 4, 'Ex' : 5},
                 'CentralAir' : {'N' : 0, 'Y' : 1},
                 'KitchenQual' : {'Po' : 1, 'Fa' : 2, 'TA' : 3, 'Gd' : 4, 'Ex' : 5},
                 'FireplaceQu' : {'NoFireplc' : 0, 'Po' : 1, 'Fa' : 2, 'TA' : 3, 'Gd' : 4, 'Ex' : 5},
                 'GarageFinish' : {'NoGarage' : 0, 'Unf' : 1, 'RFn' : 2, 'Fin' : 3},
                 'GarageQual' : {'NoGarage' : 0, 'Po' : 1, 'Fa' : 2, 'TA' : 3, 'Gd' : 4, 'Ex' : 5},
                 'GarageCond' : {'NoGarage' : 0, 'Po' : 1, 'Fa' : 2, 'TA' : 3, 'Gd' : 4, 'Ex' : 5},
                 'PavedDrive' : {'N' : 0, 'P' : 0.5, 'Y' : 1},
                 
                })

print('Binning our continuous variables into categories ...')
df.YearRemodAdd = pd.cut(df.YearRemodAdd,
                         [1950, 1970, 1990, df.YearRemodAdd.nlargest(n=1)],
                         labels=['VeryOldRemod', 'OldRemod', 'RecentRemod'],
                         include_lowest=True)

df.MasVnrArea = pd.cut(df.MasVnrArea,
                       [-1, 0, 500, 1000, df.MasVnrArea.nlargest(n=1)],
                       labels=['NoVnr', 'VnrArea_Small', 'VnrArea_Medium', 'VnrArea_Large'])

df.BsmtFinSF1 = pd.cut(df.BsmtFinSF1,
                       [-1, 0, 500, 1000, 1500, df.BsmtFinSF1.nlargest(n=1)],
                       labels=['NoBsmt', 'BsmtSF_Small', 'BsmtSF_Medium', 'BsmtSF_Large', 'BsmtSF_ExtraLarge'])

df['2ndFlrSF'] = pd.cut(df['2ndFlrSF'],
                        [-1, 0, 500, 1000, 1500, df['2ndFlrSF'].nlargest(n=1)],
                        labels=['NoSecondFlr', 'SecondFlr_Small', 'SecondFlr_Medium', 'SecondFlr_Large', 'SecondFlr_ExtraLarge'])

df.WoodDeckSF = pd.cut(df.WoodDeckSF,
                       [-1, 0, 200, 400, df.WoodDeckSF.nlargest(n=1)],
                       labels=['NoDeck', 'WoodDeckSF_Small', 'WoodDeckSF_Medium', 'WoodDeckSF_Large'])



print('Changing features to their correct data types ...')
df.BsmtCond = df.BsmtCond.astype('int64')
df.BsmtFinType2 = df.BsmtFinType2.astype('int64')
df.FireplaceQu = df.FireplaceQu.astype('int64')
print('All variables have the correct data types!')


### Feature Transformations ###
print('Taking the natural logarithm for SalePrice as the target variable ...')
df['LogSalePrice'] = np.log1p(df.SalePrice)

print('Logarithmic transformation for other numerical variables ...')

df['LogLotFrontage'] = np.log1p(df.LotFrontage)

df['LogLotArea'] = np.log1p(df.LotArea)

df['LogFirstFlrSF'] = np.log1p(df['1stFlrSF'])

df['LogGroundArea'] = np.log(df.GrLivArea)

print('Variable transformations complete.')

### Dimension reduction by simplifying categories within categorical variables ### 
print('Simplifying categories within categorical variables ...')

df.LotShape = df.LotShape.apply(lambda x:
                                0 if x == 'Reg'
                                else 1)

df.LandSlope = df.LandSlope.apply(lambda x:
                                  0 if x == 'Gtl'
                                  else 1)

df.LotConfig = df.LotConfig.apply(lambda x:
                                  'Frontage' if x == 'FR2'
                                  else 'Frontage' if x == 'FR3'
                                  else x)

df.Condition1 = df.Condition1.apply(lambda x:
                                    x if x == 'Norm'
                                    else 'MainRoad' if x == 'Feedr'
                                    else 'MainRoad' if x == 'Artery'
                                    else 'MainRoad' if x == 'PosN'
                                    else 'MainRoad' if x == 'PosA'
                                    else 'RailRoad')

df.RoofStyle = df.RoofStyle.apply(lambda x:
                                  1 if x == 'Hip'
                                  else 0)

df.Exterior1st = df.Exterior1st.apply(lambda x:
                                      'Other' if x == 'BrkComm'
                                      else 'Other' if x == 'AsphShn'
                                      else 'Other' if x == 'Stone'
                                      else 'Other' if x == 'ImStucc'
                                      else 'Other' if x == 'CBlock'
                                      else x)

df.Exterior2nd = df.Exterior2nd.apply(lambda x:
                                      'Other' if x == 'BrkComm'
                                      else 'Other' if x == 'AsphShn'
                                      else 'Other' if x == 'Stone'
                                      else 'Other' if x == 'ImStucc'
                                      else 'Other' if x == 'CBlock'
                                      else x)

df.Foundation = df.Foundation.apply(lambda x:
                                    'Other' if x == 'Wood'
                                    else 'Other' if x == 'Slab'
                                    else 'Other' if x == 'Stone'
                                    else x)

df.GarageType = df.GarageType.apply(lambda x:
                                    'Other' if x == 'CarPort'
                                    else 'Other' if x == 'Basment'
                                    else 'Other' if x == '2Types'
                                    else x)

df.Electrical = df.Electrical.apply(lambda x:
                                    1 if x == 'SBrkr'
                                    else 0)

df.Functional = df.Functional.apply(lambda x:
                                    1 if x == 'Typ'
                                    else 0)

df.SaleType = df.SaleType.apply(lambda x :
                                x if x == 'WD'
                                else 'WD' if x == 'CWD'
                                else 'WD' if x == 'VWD'
                                else 'New' if x == 'New'
                                else 'COD' if x == 'COD'
                                else 'Other')

df.SaleCondition = df.SaleCondition.apply(lambda x:
                                          'Other' if x == 'AdjLand'
                                          else 'Other' if x == 'Alloca'
                                          else 'Other' if x == 'Family'
                                          else x)

print('Categorical variables have been simplified.')

# Dropping variables that have been deemed to be unecessary after preprocessing
vars_to_drop = ['Id', 'PoolArea', 'GarageYrBlt', 'BsmtFinSF2', 'SalePrice', 'LotFrontage', 'LotArea', 'GrLivArea',
				'1stFlrSF', 'LowQualFinSF', 'OpenPorchSF', 'EnclosedPorch', '3SsnPorch', 'ScreenPorch', 'Street',
				'Utilities', 'Condition2', 'RoofMatl', 'Heating', 'Fireplaces', 'GarageArea', 'GarageCond']

print('The following {} variables will be dropped : {}'.format(len(vars_to_drop), vars_to_drop))

df = df.drop(vars_to_drop, axis=1)

print('Preprocssing Complete!\nOur cleaned dataset has {} rows and {} columns.'.format(df.shape[0], df.shape[1]))

file_out = 'cleaned_dataset.pickle'
with open(file_out, 'wb') as f:
    pickle.dump(df, f)

print('The cleaned dataset has been saved as "{}" in the local drive.'.format(file_out))