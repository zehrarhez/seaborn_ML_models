import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from scipy.stats import pearsonr
warnings.filterwarnings('ignore')

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)

df = sns.load_dataset('exercise')
print(df.head())

print(df.shape)
print(df.nunique())
print(df.info())
df.drop('Unnamed: 0', axis=1, inplace= True)

print(df.describe(include='all').T)

print(df.isnull().sum())
# No null values

def grab_col_names(dataframe, cat_th=10, car_th=20):
    cat_col= [col for col in dataframe.columns if dataframe[col].dtypes == 'O']
    num_but_cat = [col for col in dataframe.columns if dataframe[col].dtypes != 'O' and dataframe[col].nunique() < cat_th]
    cat_but_car = [col for col in dataframe.columns if dataframe[col].dtypes == 'O' and dataframe[col].nunique() > car_th]
    cat_col= cat_col + num_but_cat
    cat_col = [col for col in cat_col if col not in cat_but_car]

    #Num cols
    num_col= [col for col in dataframe.columns if dataframe[col].dtypes != 'O']
    num_col = [col for col in num_col if col not in num_but_cat]

    return cat_col, num_col

cat_col, num_col = grab_col_names(df)
print(f'Categoric columns: {cat_col}')
print(f'Numeric columns: {num_col}')

# Şekil ve alt grafikler oluştur
fig, axes = plt.subplots(nrows=1, ncols=4, figsize=(20,10))
plt.subplots_adjust(wspace=0.4)  # Alt grafikler arasındaki yatay boşluğu ayarla

def plot_box_and_line(ax, x, y, title):
    sns.boxplot(x=x, y=y, data=df, ax=ax)
    ax.set_title(title)
    mean_pulse_by_group = df.groupby(x)[y].mean().reset_index()
    median_pulse_by_group = df.groupby(x)[y].median().reset_index()
    sns.lineplot(x=x, y=y, data=mean_pulse_by_group, ax=ax, marker="o", color='red', label='Mean')
    sns.lineplot(x=x, y=y, data=median_pulse_by_group, ax=ax, marker="o", color='blue', label='Median')
    ax.set_xlabel(x)
    ax.set_ylabel(y)
    for index, row in mean_pulse_by_group.iterrows():
        ax.text(index, row[y], f'{row[y]:.2f}', color='black', ha="center", va="bottom")
    for index, row in median_pulse_by_group.iterrows():
        ax.text(index, row[y], f'{row[y]:.2f}', color='black', ha="center", va="top")
    ax.lines[-1].set_linestyle("--")  # Median ve mean çizgileri arasındaki çizgiyi kesikli yap

def plot_box_and_line_with_hue(ax, x, y, title, hue=None):
    sns.boxplot(x=x, y=y, hue=hue, data=df, ax=ax)
    ax.set_title(title)

    unique_hue_values = df[hue].unique()

    for hue_value in unique_hue_values:
        subset = df[df[hue] == hue_value]
        mean_pulse_by_group = subset.groupby(x)[y].mean().reset_index()
        median_pulse_by_group = subset.groupby(x)[y].median().reset_index()
        sns.lineplot(x=x, y=y, data=mean_pulse_by_group, ax=ax, marker="o", color='red', label=f'Mean ({hue_value})')
        sns.lineplot(x=x, y=y, data=median_pulse_by_group, ax=ax, marker="o", color='blue', label=f'Median ({hue_value})')

        for index, row in mean_pulse_by_group.iterrows():
            ax.text(index, row[y], f'{row[y]:.2f}', color='black', ha="center", va="bottom")
        for index, row in median_pulse_by_group.iterrows():
            ax.text(index, row[y], f'{row[y]:.2f}', color='black', ha="center", va="top")

    ax.set_xlabel(x)
    ax.set_ylabel(y)
    ax.legend()  # Efsaneyi eklemeyi unutmayın


# Alt grafik 1
plot_box_and_line(axes[0], 'kind', 'pulse', 'Kind ve Pulse İlişkisi')

# Alt grafik 2
plot_box_and_line(axes[1], 'diet', 'pulse', 'Diet ve Pulse İlişkisi')

# Alt grafik 3
plot_box_and_line(axes[2], 'time', 'pulse', 'Time ve Pulse İlişkisi')

# Alt grafik 4
plot_box_and_line_with_hue(axes[3], 'kind', 'pulse', 'Kind ve Pulse İlişkisi', hue='diet')
plt.legend()
plt.show()

#outlier detection
def outlier_thresholds(dataframe, col_name, q1=0.25, q3=0.75):
    quartile1 = dataframe[col_name].quantile(q1)
    quartile3 = dataframe[col_name].quantile(q3)
    interquantile_range = quartile3 - quartile1
    up_limit = quartile3 + 1.5 * interquantile_range
    low_limit = quartile1 - 1.5 * interquantile_range
    return low_limit, up_limit

def outlier_detection():
    df_copy = df.copy()  # Veri çerçevesinin kopyasını oluşturuyoruz
    for col in num_col:
        low_limit, up_limit = outlier_thresholds(df_copy, col)
        df_copy.loc[(df_copy[col] < low_limit) | (df_copy[col] > up_limit), 'outlier'] = col
    return df_copy['outlier'].value_counts()

outlier_detection()
#bu outlierlar gerçek verileri yansıtıyor, o yüzden herhangi bir işlem yapmıyoruz.

# Kategorik sütunları encode etmek

for col in cat_col:
    unique_values = df[col].unique()
    mapping = {value: i for i, value in enumerate(unique_values)}
    df[col] = df[col].replace(mapping)
    print(f"{col} Mapping: {mapping}")


plt.figure(figsize=(10, 8))
sns.heatmap(df.corr(), annot=True, cmap='coolwarm', center=0)
plt.title('Correlation Matrix')
plt.show()


print("------------------------------------------------------------------------------------------------")
# H0: There is no significant correlation
# H1: There is a significant correlation

correlation_coefficient, p_value = pearsonr(df['pulse'], df['kind'])
print("Correlation Coefficient:", correlation_coefficient)
print("P-value:", p_value)

alpha = 0.05  # Set the significance level
if p_value < alpha:
    print("Reject the null hypothesis: There is a significant correlation.")
    print("Dataset is suitable for linear regression.")
    print("Dataset is stationary.")
else:
    print("Fail to reject the null hypothesis: There is no significant correlation.")
    print("Dataset is not suitable for linear regression.")
    print("Dataset is not stationary.")



# Pulse değerlerini aralıklara dönüştürme
def categorize_pulse(pulse_value):
    if pulse_value < 110:
        return f"Pulse Value : {pulse_value} --> Normal"
    else:
        return f"Pulse Value : {pulse_value} --> High"

df['pulse_category'] = df['pulse'].apply(categorize_pulse)

# Bağımsız değişkenler ve hedef değişken ayırma
X = df[['kind', 'time', 'diet']]
y = df['pulse_category']

# Veriyi eğitim ve test olarak bölmek
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.21, random_state=42)

# Modeli eğitme
model = RandomForestClassifier()
model.fit(X_train, y_train)

# Modeli kullanarak tahmin yapma
y_pred = model.predict(X_test)

kind_values = df['kind'].value_counts()
time_values = df['time'].value_counts()
diet_values = df['diet'].value_counts()

print("Kind Values:", kind_values)
print("Time Values:", time_values)
print("Diet Values:", diet_values)

#örnek tahmin
prediction = model.predict([[0, 1, 1]])
print("Prediction:", prediction)

prediction2 = model.predict([[2, 2, 1]])
print("Prediction2:", prediction2)

import pickle
def save_model_to_pickle(model, filename):
    with open(filename, 'wb') as file:
        pickle.dump(model, file)
    print(f"Model is saved to {filename}")

# Replace 'best_model.pkl' with the desired filename for your saved model
save_model_to_pickle(model, 'model.pkl')

def prediction_from_user():
    # Kullanıcıdan veri girişi alma
    user_kind = int(input("Kind (0, 1, 2): "))
    user_time = int(input("Time (0, 1, 2): "))
    user_diet = int(input("Diet (0, 1): "))

    # Kullanıcının girdiği verileri kullanarak tahmin yapma
    user_input = np.array([[user_kind, user_time, user_diet]])
    predicted_category = model.predict(user_input)[0]

    print("Predicted Pulse Category:", predicted_category)
