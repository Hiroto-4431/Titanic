# ライブラリ読み込み
import numpy as np
import pandas as pd
from pandas import DataFrame, Series
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

import warnings
warnings.filterwarnings('ignore')


# データ読み込み
path = '/Users/hiroto/PycharmProjects/Titanic/data/'
df = pd.read_csv(path + 'train.csv')
df_test = pd.read_csv(path + 'test.csv')


# データの概観
print('-----【データの全体像】-----')
print('Train Data = データ数:{}, 変数:{}種類'.format(df.shape[0], df.shape[1]))
print('Test Data = データ数:{}, 変数:{}種類'.format(df_test.shape[0], df.shape[1]))

print('\n -----【データ】-----')
print(df.head(3))

print('\n -----【データのカラム一覧】-----')
print(df.columns)


# データの分析
# 欠損値の確認
print('\n　-----【Train Data : 欠損値】----- \n', df.isnull().sum())
print('\n　-----【Test Data : 欠損値】----- \n', df_test.isnull().sum())


# EDA
# 生存者の割合
f, ax = plt.subplots(1, 2, figsize=(18, 8))
value_counts = df['Perished'].value_counts()
# グラフ1
labels = value_counts.index.to_series().replace({0:'alive', 1: 'dead'})
colors = value_counts.index.to_series().replace({0:'#E6B422', 1:'#C0C0C0'})
df['Perished'].value_counts().plot.pie(explode=[0, 0.1], autopct='%1.1f%%', ax=ax[0], labels=labels, colors=colors, shadow=True, startangle=90)
ax[0].set_title('Perished', fontsize=14)
ax[0].set_ylabel('')
# グラフ2
sns.countplot(x='Perished', data=df, ax=ax[1], palette={'0':'#E6B422', '1':'#C0C0C0'})
plt.xticks([0, 1], ['alive', 'dead'])
ax[1].set_title('Perished', fontsize=14)
plt.show()

# 年齢
# Age（年齢） EDA
f, ax = plt.subplots(1, 2, figsize=(18, 8))
palette = {0:'#E6B422', 1:'#C0C0C0'}
# グラフ1
sns.violinplot(x="Pclass", y="Age", hue="Perished", data=df, split=True, ax=ax[0], palette=palette)
ax[0].set_title('Pclass and Age vs Perished', fontsize=14)
ax[0].set_yticks(range(0, 110, 10))
# グラフ2
sns.violinplot(x="Sex", y="Age", hue="Perished", data=df, split=True, ax=ax[1], palette=palette)
ax[1].set_title('Sex and Age vs Perished', fontsize=14)
ax[1].set_yticks(range(0, 110, 10))
plt.show()


# データ前処理
# 'Age', 'Fre', 'Cabin', 'Embarked' カラムを削除
missing_list = ['Age', 'Fare', 'Cabin', 'Embarked']
df.drop(missing_list, axis=1, inplace=True)
df_test.drop(missing_list, axis=1, inplace=True)

# カテゴリカル変数の処理
# 'Name', 'Sex', 'Ticket' カラムを削除
category_list = ['Name', 'Sex', 'Ticket']
df.drop(category_list, axis=1, inplace=True)
df_test.drop(category_list, axis=1, inplace=True)


# 機械学習モデルの構築・学習
# ロジスティクス回帰の構築
X = df.iloc[:, 2:].values  # X = 説明変数
y = df.iloc[:, 1].values  # y = 目的変数
X_test = df_test.iloc[:, 1:].values


# データ分割
# ホールドアウト法
X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.3, random_state=42)


# ロジスティクス回帰モデルの作成
lr = LogisticRegression(random_state=42)
lr.fit(X_train, y_train)


# 精度
print('\n　-----【精度】-----')
print('Train Score : {}'.format(round(lr.score(X_train, y_train), 3)))
print('Test Score : {}'.format(round(lr.score(X_valid, y_valid), 3)))
