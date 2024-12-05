#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd 


# In[25]:


df = pd.read_csv('experiment_lesson_4.csv')
df


# In[27]:


df_test = df[df['experiment_group'] == 'test']
df_control = df[df['experiment_group'] == 'control']


# In[35]:


df_test_grouped = df_test.groupby('district')
df_test_grouped


# In[34]:


import matplotlib.pyplot as plt


# In[39]:


for district, data in df_test_grouped:
    plt.figure()  # Создание новой фигуры для каждого района
    data['delivery_time'].hist(bins=10)  # Построение гистограммы
    plt.title(f'Гистограмма для района {district}')
    plt.xlabel('Название переменной')
    plt.ylabel('Частота')
    plt.show()


# In[40]:


df_control_grouped = df_control.groupby('district')
for district, data in df_control_grouped:
    plt.figure()  # Создание новой фигуры для каждого района
    data['delivery_time'].hist(bins=10)  # Построение гистограммы
    plt.title(f'Гистограмма для района {district}')
    plt.xlabel('Название переменной')
    plt.ylabel('Частота')
    plt.show()


# In[41]:


df.groupby('experiment_group').agg({'order_id':'count'})


# In[49]:


delivery_time_test = df.query("experiment_group == 'test'").delivery_time

#и кладём в функцию
scipy.stats.normaltest(delivery_time_test)


# In[46]:


from scipy import stats  # For statistical functions
from scipy import optimize  # For optimization functions
import scipy


# In[50]:


delivery_time_control = df.query("experiment_group == 'control'").delivery_time

#и кладём в функцию
scipy.stats.normaltest(delivery_time_control)


# In[52]:


std_test = np.std(delivery_time_test)
std_test


# In[53]:


std_control = np.std(delivery_time_control)
std_control


# In[57]:


scipy.stats.ttest_ind(delivery_time_test, delivery_time_control)


# In[59]:


test_mean = delivery_time_test.mean()


# In[60]:


control_mean = delivery_time_control.mean()


# In[67]:


round((test_mean - control_mean)/control_mean, 4)


# In[65]:


test_mean


# In[66]:


control_mean


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[3]:


df['timestamp'] = pd.to_datetime(df['timestamp'])
df


# In[4]:


import seaborn as sns


# In[5]:


sns.lineplot(data=df, x='timestamp', y='cnt')


# In[6]:


df.set_index('timestamp', inplace=True)

# Агрегация данных по дням (число поездок суммируется по каждому дню)
daily_data = df.resample('D').cnt.sum()


# In[7]:


plt.figure(figsize=(10, 6))
daily_data.plot(kind='line', title='Агрегация числа поездок по дням')
plt.xlabel('Дата')
plt.ylabel('Число поездок')
plt.grid(True)
plt.show()


# In[8]:


# Скользящее среднее с окном 3
rolling_mean = daily_data.rolling(window=3).mean()
rolling_mean


# In[9]:


difference = daily_data - rolling_mean

# Вывод разницы
print(difference)


# In[10]:


import numpy as np


# In[11]:


std = np.std(difference)


# In[12]:


std


# In[17]:


daily_data= daily_data.to_frame().reset_index()
daily_data


# In[13]:


rolling_mean = rolling_mean.to_frame().reset_index()


# In[14]:


rolling_mean['upper_bound'] = rolling_mean['cnt'] + 2.576*std


# In[15]:


rolling_mean['lower_bound'] = rolling_mean['cnt'] - 2.576*std


# In[16]:


rolling_mean


# In[18]:


new_df = daily_data.merge(rolling_mean, on='timestamp')


# In[19]:


new_df


# In[24]:


daily_data


# In[22]:


# Нахождение аномально высоких значений (наблюдаемые значения > верхней границы)
anomalies_high = new_df[new_df['cnt_x'] > new_df['upper_bound']]

# Вывод аномально высоких значений
anomalies_high


# In[24]:


# Нахождение аномально высоких значений (наблюдаемые значения > верхней границы)
anomalies_fail = new_df[new_df['cnt_x'] < new_df['lower_bound']]

# Вывод аномально высоких значений
anomalies_fail


# In[42]:


import numpy as np


# In[45]:


grouped_df = df.groupby('fb_campaign_id')['Impressions'].sum().reset_index()

# Логарифмирование значений показов
grouped_df['log_impressions'] = np.log(grouped_df['Impressions'])

# Построение гистограммы
plt.figure(figsize=(10, 6))
sns.histplot(grouped_df['log_impressions'], kde=False, bins=10, color='skyblue')

# Настройка графика
plt.xlabel('Log(Impressions)')
plt.ylabel('Frequency')
plt.title('Распределение числа показов (логарифмированные значения)')

# Показать график
plt.show()


# In[69]:


df['ctr'] = df['Clicks']/df['Impressions']
df


# In[52]:


df.sort_values(by='ctr').


# In[51]:


df[df['ctr'] =='0.001059']['ad_id'].value_counts()


# In[54]:


# Анализ CTR для кампании с ID 916
campaign_916_df = df[df['xyz_campaign_id'] == 916]

# Построение гистограммы распределения CTR для кампании 916
plt.figure(figsize=(12, 6))
sns.histplot(campaign_916_df['ctr'], kde=False, bins=10, color='lightgreen')
plt.xlabel('CTR')
plt.ylabel('Frequency')
plt.title('Распределение CTR для кампании 916')
plt.show()


# In[70]:


df['cpc'] = df['Spent']/df['Clicks']
df['cpc'].describe()


# In[57]:


df = df.dropna(subset=['cpc'])


# In[71]:


from scipy.stats import iqr
cpc_iqr = iqr(df['cpc'], nan_policy='omit')
print(f"Межквартильный размах (IQR) округленный до двух знаков после точки: {cpc_iqr:.2f}")


# In[72]:


plt.figure(figsize=(12, 6))
sns.histplot(df['cpc'], kde=False, bins=10, color='skyblue')
plt.xlabel('cpc (Стоимость за клик)')
plt.ylabel('Частота')
plt.title('Распределение CPC по кампаниям')
plt.show()


# In[80]:


# Гистограмма CPC с использованием distplot (устаревшая версия)
plt.figure(figsize=(12, 6))
sns.distplot(df.dropna(subset=['cpc'])['cpc'], kde=False, hist=True, label='All Data')
sns.distplot(df.dropna(subset=['cpc']).loc[df['gender'] == 'Male', 'cpc'], kde=False, hist=True, label='Male')
sns.distplot(df.dropna(subset=['cpc']).loc[df['gender'] == 'Female', 'cpc'], kde=False, hist=True, label='Female')

# Настройка графика
plt.xlabel('CPC (Стоимость за клик)')
plt.ylabel('Частота')
plt.title('Распределение CPC по полу пользователей')
plt.legend()
plt.show()


# In[81]:


df['cr'] = df['Approved_Conversion']/df['Clicks']


# In[85]:


df[df['ad_id']==1121814]['cr']


# In[ ]:





# In[ ]:





# In[ ]:





# In[7]:


df.isna().sum()


# In[8]:


df['Year'].describe()


# In[9]:


df['Year'].mode()


# In[14]:


df['Platform'].value_counts(normalize=True)


# In[26]:


df=df[df['Year'].notna()]


# In[27]:


df


# In[28]:


df['Publisher'].value_counts()


# In[30]:


df[df['Publisher']=='Nintendo'].median()


# In[33]:


df_nintendo = df[df['Publisher']=='Nintendo']


# In[34]:


import seaborn as sns
import matplotlib.pyplot as plt

# Загрузка данных (предположим, данные хранятся в csv-файле)
# df = pd.read_csv('nintendo_sales.csv')

# Увеличим размер графика
plt.figure(figsize=(16, 16))

# Построение боксплота
sns.boxplot(x='Genre', y='JP_Sales', data=df_nintendo)

# Показать график
plt.xticks(rotation=90)  # Поворот подписей жанров для удобства чтения
plt.show()


# In[35]:


filtered_genres = ['Fighting', 'Simulation', 'Platform', 'Racing', 'Sports']
filtered_df = df[df['Genre'].isin(filtered_genres)]

# Группируем данные по жанру и году выпуска, суммируем глобальные продажи
sales_by_year_genre = filtered_df.groupby(['Year', 'Genre'])['Global_Sales'].sum().reset_index()

# Визуализируем динамику изменения продаж по жанрам
plt.figure(figsize=(14, 8))
sns.lineplot(data=sales_by_year_genre, x='Year', y='Global_Sales', hue='Genre')

# Добавим подписи и легенду
plt.title('Динамика мировых продаж игр Nintendo по жанрам')
plt.xlabel('Год')
plt.ylabel('Объем мировых продаж (млн копий)')
plt.legend(title='Жанры', loc='upper right')
plt.show()


# In[ ]:




