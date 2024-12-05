#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd


# In[2]:


df = pd.read_csv('5_task_1.csv')
df


# In[6]:


pip install pingouin


# In[7]:


import pingouin as pg


# In[8]:


pg.homoscedasticity(df, dv='events', group='group', method='levene')


# In[16]:


stat, p_value = scipy.stats.normaltest(df.events)

print(f"Statistics: {stat}, p-value: {p_value}")


# In[18]:


import scipy.stats as stats
import matplotlib.pyplot as plt
import seaborn as sns


# In[22]:


# Разделение данных по группам
group_A = df[df['group'] == 'A']['events']
group_B = df[df['group'] == 'B']['events']
group_C = df[df['group'] == 'C']['events']

# 1. Проверка на нормальность с помощью теста D'Agostino
stat_A, p_value_A = stats.normaltest(group_A)
stat_B, p_value_B = stats.normaltest(group_B)
stat_C, p_value_C = stats.normaltest(group_C)

print(f"Group A - Statistics: {stat_A}, p-value: {p_value_A}")
print(f"Group B - Statistics: {stat_B}, p-value: {p_value_B}")
print(f"Group C - Statistics: {stat_C}, p-value: {p_value_C}")


# In[31]:


import statsmodels.formula.api as smf 
from statsmodels.stats.anova import anova_lm
model = smf.ols(formula = "events ~ group", data = df).fit() 
anova_lm(model)


# In[32]:


from statsmodels.stats.multicomp import (pairwise_tukeyhsd,
                                         MultiComparison)


# In[35]:


pg.pairwise_tukey(data=df, dv="events", between="group")


# In[36]:


print(pairwise_tukeyhsd(df.events, df.group))


# In[37]:


df = pd.read_csv('5_task_2.csv')
df


# In[39]:


test = df[df['group'] == 'test']


# In[41]:


control = df[df['group'] == 'control']


# In[40]:


plt.figure(figsize=(10, 6))
sns.histplot(data=test, x='events', hue='group', kde=True, bins=10)

# Настройки графика
plt.title('Распределение событий для контрольной и тестовой группы')
plt.xlabel('Количество событий')
plt.ylabel('Частота')
plt.legend(title='Группа')
plt.show()


# In[42]:


plt.figure(figsize=(10, 6))
sns.histplot(data=control, x='events', hue='group', kde=True, bins=10)

# Настройки графика
plt.title('Распределение событий для контрольной и тестовой группы')
plt.xlabel('Количество событий')
plt.ylabel('Частота')
plt.legend(title='Группа')
plt.show()


# In[48]:


control.groupby('segment')['events'].describe().T


# In[50]:


model = smf.ols(formula = "events ~ segment + group + segment:group", data = df).fit() 
anova_lm(model, typ = 2)


# In[53]:


df['combination'] = df['segment'] + ' \ ' + df['group']


# In[54]:


pairwise_tukeyhsd(df.events, df.combination).summary()


# In[57]:


sns.pointplot(data=df,  x='events', y='combination', hue='segment')


# In[ ]:




