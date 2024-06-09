# ライブラリ読み込み
import numpy as np
import pandas as pd
from pandas import DataFrame, Series
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score

import warnings
warnings.filterwarnings('ignore')


# データ読み込み
path = '/Users/hiroto/PycharmProjects/Titanic/data/'
df = pd.read_csv(path + 'train.csv')
df_test = pd.read_csv(path + 'test.csv')


# データの概観
print('-----【データの全体像】-----')
print('Train Data = データ数:{}, 変数:{}種類'.format(df.shape[0], df.shape[1]))
print('Test Data = データ数:{}, 変数:{}種類'.format(df_test.shape[0], df.shape[1]), '\n')

print('-----【データ】-----')
print(df.head(3), '\n')

print('-----【データのカラム一覧】-----')
print(df.columns, '\n')


# データの分析
# 欠損値の確認
print('-----【Train Data : 欠損値】-----')
print(df.isnull().sum(), '\n')

print('-----【Test Data : 欠損値】-----')
print(df_test.isnull().sum(), '\n')

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

# 相関図
df_numeric = df.select_dtypes(include=['number'])
sns.heatmap(df_numeric.corr(), annot=True, cmap='PuBu')
fig = plt.gcf()
fig.set_size_inches(10, 8)
plt.show()


# データ前処理
# 'Age', 'Fare' 平均値で補完
age = pd.concat([df['Age'], df_test['Age']])
fare = pd.concat([df['Fare'], df_test['Fare']])

df['Age'].fillna(age.mean(), inplace=True)
df_test['Age'].fillna(age.mean(), inplace=True)

df['Fare'].fillna(fare.mean(), inplace=True)
df_test['Fare'].fillna(fare.mean(), inplace=True)

print('-----【Age, Fare, 欠損値対応後】-----')
print(df.isnull().sum(), '\n')

# Cabin カラムを削除
df.drop('Cabin', axis=1, inplace=True)
df_test.drop('Cabin', axis=1, inplace=True)

print('-----【欠損値】-----')
print(df.isnull().sum(), '\n')

# Embarked 欠損値はSで対応
sns.countplot(x='Embarked', data=df, palette={'S':'#E6B422', 'C':'#C0C0C0', 'Q':'#B87333'})
plt.title('NUmber of Passengers Boarded')
plt.show()

df['Embarked'].fillna('S', inplace=True)
df_test['Embarked'].fillna('S', inplace=True)

print('-----【Embarked, 欠損値対応後】-----')
print(df.isnull().sum(), '\n')


# カテゴリカル変数の処理
# 'Name', 'Sex', 'Ticket' カラムを削除
df.drop('Name', axis=1, inplace=True)
df_test.drop('Name', axis=1, inplace=True)

df.drop('Ticket', axis=1, inplace=True)
df_test.drop('Ticket', axis=1, inplace=True)

print('-----【Name, Sex, Ticket, カラム削除後のカラム一覧】-----')
print(df.columns, '\n')

# 'Sex' カラムを0, 1で置き換える
df.replace({'Sex': {'male': 0, 'female': 1}}, inplace=True)
df_test.replace({'Sex': {'male': 0, 'female': 1}}, inplace=True)

print('-----【Sex カラムを0, 1で置き換えた後のデータ】-----')
print(df.head(), '\n')

# Embarked One-Hot Encoding処理
embarked = pd.concat([df['Embarked'], df_test['Embarked']])

embarked_ohe = pd.get_dummies(embarked)

embarked_ohe_train = embarked_ohe[:891]
embarked_ohe_test = embarked_ohe[891:]

df = pd.concat([df, embarked_ohe_train], axis=1)
df_test = pd.concat([df_test, embarked_ohe_test], axis=1)

df.drop('Embarked', axis=1, inplace=True)
df_test.drop('Embarked', axis=1, inplace=True)

print('-----【Embarked One-Hot Encoding処理後のデータ】-----')
print(df.head(), '\n')


# ベースラインモデルの構築
X = df.iloc[:, 2:].values  # X = 説明変数
y = df.iloc[:, 1].values  # y = 目的変数
X_test = df_test.iloc[:, 1:].values


# データ分割
# ホールドアウト法
X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.3, random_state=42)


# ランダムフォレストモデルの作成
rfc = RandomForestClassifier(max_depth=10, min_samples_leaf=1, n_estimators=100, n_jobs=-1, random_state=42)
rfc.fit(X_train, y_train)


# 精度
print('-----【精度】-----')
print('Train Score : {}'.format(round(rfc.score(X_train, y_train), 3)))
print('Test Score : {}'.format(round(rfc.score(X_valid, y_valid), 3)), '\n')


# 過学習
# ハイパーパラメータの調整
# グリッドサーチ
param_grid = {'max_depth': [3, 5, 7], 'min_samples_leaf': [1, 2, 4]}

print('-----【グリッドサーチ】-----')
for max_depth in param_grid['max_depth']:
    for min_samples_leaf in param_grid['min_samples_leaf']:
        rfc_grid = RandomForestClassifier(max_depth=max_depth, min_samples_leaf=min_samples_leaf, n_estimators=100, n_jobs=-1, random_state=42)
        rfc_grid.fit(X_train, y_train)
        print('max_depth: {}, min_samples_leaf: {}'.format(max_depth, min_samples_leaf))
        print('    Train Score: {}, Test Score: {}'.format(round(rfc_grid.score(X_train, y_train), 3), round(rfc_grid.score(X_valid, y_valid), 3)))

# クロスバリデーション
rfc_gs = GridSearchCV(RandomForestClassifier(n_estimators=100, n_jobs=-1, random_state=42), param_grid, cv=5)
rfc_gs.fit(X, y)

print('\n')
print('-----【クロスバリデーション】-----')
print('Best Parameters: {}'.format(rfc_gs.best_params_))
print('CV Score: {}'.format(round(rfc_gs.best_score_, 3)), '\n')


# 特徴量エンジニアリング
df_fe = df.copy()
df_fe_test = df_test.copy()

df_fe['Family'] = df['SibSp'] + df['Parch']
df_fe_test['Family'] = df_test['SibSp'] + df_test['Parch']

print('-----【SibSp, Parch, の値の和をとって、Familyという新しい変数を加える】-----')
print(df_fe.head(), '\n')

# 新たなデータを、ベースモデルと同様の構成のランダムフォレストに学習させてみる
X_fe = df_fe.iloc[:, 2:].values
y_fe = df_fe.iloc[:, 1].values

X_fe_test = df_fe_test.iloc[:, 1:].values

X_fe_train, X_fe_valid, y_fe_train, y_fe_valid = train_test_split(X_fe, y_fe, test_size=0.3, random_state=42)

rfc_fe = RandomForestClassifier(max_depth=7, min_samples_leaf=1, n_estimators=100, n_jobs=-1, random_state=42)
rfc_fe.fit(X_fe_train, y_fe_train)

print('-----【精度】-----')
print('Train Score: {}'.format(round(rfc_fe.score(X_fe_train, y_fe_train), 3)))
print('Test Score: {}'.format(round(rfc_fe.score(X_fe_valid, y_fe_valid), 3)), '\n')


# 様々なモデルの構築・調整
# ロジスティクス回帰モデル
lr = LogisticRegression(random_state=42)
lr.fit(X_train, y_train)

print('-----【Logistic Regression】-----')
print('Train Score: {}'.format(round(lr.score(X_train, y_train), 3)))
print('Test Score: {}'.format(round(lr.score(X_valid, y_valid), 3)), '\n')

# 多層パーセプトロンモデル
mlpc = MLPClassifier(hidden_layer_sizes=(100, 100, 10), random_state=0)
mlpc.fit(X_train, y_train)

print('-----【Multilayer Perceptron】-----')
print('Train Score: {}'.format(round(mlpc.score(X_train, y_train), 3)))
print('Test Score: {}'.format(round(mlpc.score(X_valid, y_valid), 3)), '\n')


# モデルのアンサンブリング
# トレーニングデータ精度を計算
rfc_pred_train = rfc.predict_proba(X_train)
lr_pred_train = lr.predict_proba(X_train)
mlpc_pred_train = mlpc.predict_proba(X_train)

pred_proba_train = (rfc_pred_train + lr_pred_train + mlpc_pred_train) / 3
pred_train = pred_proba_train.argmax(axis=1)

print('-----【精度】-----')
accuracy_train = accuracy_score(y_train, pred_train)
print('Train Data : {}'.format(round(accuracy_train, 3)))

# テストデータ精度を計算
rfc_pred_valid = rfc.predict_proba(X_valid)
lr_pred_valid = lr.predict_proba(X_valid)
mlpc_pred_valid = mlpc.predict_proba(X_valid)

pred_proba_valid = (rfc_pred_valid + lr_pred_valid + mlpc_pred_valid) / 3
pred_valid = pred_proba_valid.argmax(axis=1)

accuracy = accuracy_score(y_valid, pred_valid)
print('Test Data : {}'.format(round(accuracy, 3)))

# 出力すべき予測値
rfc_pred = rfc.predict_proba(X_test)
lr_pred = lr.predict_proba(X_test)
mlpc_pred = mlpc.predict_proba(X_test)

pred_proba = (rfc_pred + lr_pred + mlpc_pred) / 3
pred = pred_proba.argmax(axis=1)