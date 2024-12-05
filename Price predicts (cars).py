#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd


# In[11]:


df = pd.read_csv('cars.csv')
df


# In[12]:


df['company'] = df['CarName'].str.split(' ').apply(lambda x: x[0] if len(x) > 0 else None)


# In[13]:


df = df.drop(columns=['CarName', 'car_ID'])


# In[14]:


df['company'].unique()


# In[15]:


df['company'] = df['company'].replace(['maxda', 'nissan', 'porcshce', 'toyouta', 'vokswagen', 'vw'],
                                     ['mazda', 'Nissan', 'porsche', 'toyota', 'volkswagen', 'volkswagen'])


# In[17]:


df = df[['company', 'fueltype', 'aspiration','carbody', 'drivewheel', 'wheelbase', 'carlength','carwidth', 'curbweight', 'enginetype', 'cylindernumber', 'enginesize', 'boreratio','horsepower', 'price']]


# In[18]:


df.corr()


# In[19]:


df_dummy = pd.get_dummies(data=df[['fueltype', 'aspiration', 'carbody', 'drivewheel', 'enginetype', 
                                   'cylindernumber']], drop_first = True)
df_dummy


# In[21]:


df_dummy_new = pd.get_dummies(data=df[['company', 'fueltype', 'aspiration','carbody', 'drivewheel', 
                                       'wheelbase', 'carlength','carwidth', 'curbweight', 'enginetype', 
                                       'cylindernumber', 'enginesize', 'boreratio','horsepower', 'price']], 
                              drop_first = True)
df_dummy_new


# In[23]:


import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split


# In[25]:


# Определяем переменные
X = df[['horsepower']]  # Независимая переменная
y = df['price']         # Зависимая переменная

# Создаем и обучаем модель
model = LinearRegression()
model.fit(X, y)

# Предсказываем цены на основе мощности
y_pred = model.predict(X)

# Рассчитываем коэффициент детерминации (R^2)
r2 = r2_score(y, y_pred)

# Выводим процент объясненной изменчивости
print(f"Процент объясненной изменчивости: {round(r2 * 100)}%")


# In[28]:


df_encoded = pd.get_dummies(df, drop_first=True)

# Разделение данных на X и y
X_full = df_encoded.drop('price', axis=1)  # Все предикторы
y = df_encoded['price']                    # Зависимая переменная

# Создание модели со всеми предикторами
X_train, X_test, y_train, y_test = train_test_split(X_full, y, test_size=0.3, random_state=42)
model_full = LinearRegression()
model_full.fit(X_train, y_train)
y_pred_full = model_full.predict(X_test)
r2_full = r2_score(y_test, y_pred_full)

# Создание модели без переменной "company" (марки машин)
X_no_company = X_full.drop([col for col in X_full.columns if 'company_' in col], axis=1)
X_train_no_company, X_test_no_company, y_train_no_company, y_test_no_company = train_test_split(X_no_company, y, test_size=0.3, random_state=42)
model_no_company = LinearRegression()
model_no_company.fit(X_train_no_company, y_train_no_company)
y_pred_no_company = model_no_company.predict(X_test_no_company)
r2_no_company = r2_score(y_test_no_company, y_pred_no_company)


# In[29]:


# Добавляем константу для модели с помощью statsmodels
X_no_company_const = sm.add_constant(X_no_company)

# Строим модель с использованием statsmodels для получения значимостей предикторов
model_sm = sm.OLS(y, X_no_company_const).fit()

# Получаем сводную таблицу с результатами регрессии
summary = model_sm.summary()
print(summary)

# 1. Процент объяснённой дисперсии (R^2)
r2_no_company = model_sm.rsquared
print(f"Выбранная модель объясняет примерно {round(r2_no_company * 100)}% дисперсии.")

# 2. Количество незначимых предикторов
p_values = model_sm.pvalues
insignificant_predictors = (p_values > 0.05).sum()
print(f"Среди предикторов {insignificant_predictors} из {len(p_values)} оказались не значимыми (p > 0.05).")

# 3. Интерпретация изменения цены при изменении horsepower
horsepower_coef = model_sm.params['horsepower']
print(f"При единичном изменении показателя horsepower, цена изменится на {horsepower_coef} (без округления).")


# In[ ]:




