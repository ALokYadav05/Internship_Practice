import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, root_mean_squared_error
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import train_test_split


def load_preprocess_encode_scale_modeltrain_metrics():
    df = pd.read_csv('insurance.csv')
    print(df.head())

    print(df.columns)

    print(df.info())

    print(df.dtypes)

    print(df.describe())  #this-gives us statistics summary

    # df["charges"] = np.log(df["charges"])

    print(df.isnull().sum())

    print(df[df.duplicated()])  #this is the duplicated-row

    df.drop_duplicates(keep='first', inplace=True)         #keep='first' delete the second occurrence of similar row

    print(df['smoker'].value_counts())

    print(df['children'].value_counts())

    print(df['region'].value_counts())   # number-of-occurrence
    print(df['region'].mode())

    # Histogram-plot for Target-column
    sns.set_style('whitegrid')
    plt.figure(figsize=(10,5))
    sns.histplot(df['charges'], bins=50, kde=True, palette='dark')
    plt.title('Charges Distribution (Target)', fontsize=20, weight='bold', color='green')
    plt.xlabel('Charges', fontsize=12)
    plt.ylabel('count', fontsize=19)
    plt.xlabel('Charges', fontsize=20, weight='bold')
    plt.show()
    plt.figure(figsize=(20,15))
    sns.pairplot(df, hue='smoker' ,palette='dark')
    plt.show()


    # this-basically shows the entire dataset comparing numerical features distribution in-one go!
    corr = df.corr(numeric_only=True) # shows-correlation of one-feature with another (only-numeric columns)
    print(corr)

    # heatmap, shows correlation
    plt.figure(figsize=(20,9))
    sns.heatmap(corr, annot=True, cmap='viridis',linewidths=0.5)
    plt.plot()

    plt.figure(figsize=(20,9))
    sns.violinplot(x='region',y='charges', data=df, palette='dark')
    plt.show()

    #  The fat-part shows the most of the data point is situated there!
    #  The thin-part shows that there are outliers, because very few data-points exists on that portion!
    df['sex'] = df['sex'].astype(object)
    df['smoker'] = df['smoker'].astype(object)
    df['region'] = df['region'].astype(object)

    # q1 = df['bmi'].quantile(0.25)
    # q3 = df['bmi'].quantile(0.75)
    # IQR = q3-q1
    #
    # lower_fence = q1-1.5*IQR
    # upper_fence = q3+1.5*IQR
    #
    # df['bmi'] = df['bmi'].clip(lower=lower_fence, upper=upper_fence)


    num_feat = [cols for cols in df.columns if df[cols].dtypes != 'object' and cols != 'charges']
    print(num_feat)

    plt.figure(figsize=(10, 5))
    sns.boxplot(data=df[num_feat])
    plt.show()

    # this-is-how we treat the outlier
    for cols in num_feat:
        q1 = df[cols].quantile(0.25)
        q3 = df[cols].quantile(0.75)
        IQR = q3-q1

        lower_fence = q1-1.5*IQR
        upper_fence = q3+1.5*IQR

        df[cols]= df[cols].clip(lower=lower_fence, upper=upper_fence)

    ## separated Numeric and categorical column
    cat_features = [cols for cols in df.columns if df[cols].dtypes == 'object']
    num_features = [cols for cols in df.columns if df[cols].dtypes != 'object' and cols != 'charges']
    print(cat_features)
    print(num_features)

    x = df.iloc[:, :-1]   #features
    y = df.iloc[:, -1]    #target

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=1, shuffle=True)

    print(x_train.shape , x_test.shape)

    ohe = OneHotEncoder(drop='first', sparse_output=False)
    x_train_cat = ohe.fit_transform(x_train[cat_features])
    x_test_cat = ohe.transform(x_test[cat_features])
    print(x_train_cat, x_test_cat)

    x_train_cat = pd.DataFrame(x_train_cat, columns=ohe.get_feature_names_out(cat_features), index=x_train.index)
    x_test_cat = pd.DataFrame(x_test_cat, columns=ohe.get_feature_names_out(cat_features), index=x_test.index)

    scaler = StandardScaler()
    x_train_num = scaler.fit_transform(x_train[num_features])
    x_test_num = scaler.transform(x_test[num_features])
    print(x_train_num, x_test_num)

    x_train_num = pd.DataFrame(x_train_num, columns=num_features , index=x_train.index)
    x_test_num = pd.DataFrame(x_test_num, columns=num_features, index=x_test.index)

    x_train = pd.concat([x_train_num,x_train_cat], axis=1)
    x_test = pd.concat([x_test_num,x_test_cat], axis=1)

    regressor = LinearRegression()
    regressor.fit(x_train, y_train)

    y_predict = regressor.predict(x_test)

    mse = mean_squared_error(y_test, y_predict)
    print(f" Mean Squared Error: {mse}")

    mae = mean_absolute_error(y_test, y_predict)
    print(f"Mean Absolute Error: {mae}")

    r2 = r2_score(y_test, y_predict)
    print(f"R2 Score: {r2}")

    r_mse = root_mean_squared_error(y_test, y_predict)
    print(r_mse)


    plt.figure(figsize=(14,8))
    plt.scatter(y_test, y_predict)
    plt.xlabel("Actual Charges")
    plt.ylabel("Predicted Charges")
    plt.title("Actual and Predicted Insurance Charges")
    plt.plot(
        [y_test.min(), y_test.max()],
        [y_test.min(), y_test.max()],  color='red', linestyle='dashed', marker='o',
    )
    plt.show()


load_preprocess_encode_scale_modeltrain_metrics()