#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import math
import matplotlib.pyplot as plt
df=pd.read_csv("E:\sem_3\class\data warehouse and data mining\water_potability.csv")


# In[2]:


df.head()


# In[3]:


df.info()


# In[4]:


df


# In[5]:


df.isnull()#to cherck the nan values as it returns nan values as true


# In[6]:


df.ph=df.ph.fillna(0)#filling the nan values with o as nan values are problematic during coding


# In[7]:


df


# In[8]:


df.Hardness=df.Hardness.fillna(0)


# In[9]:


df.Solids =df.Solids.fillna(0)


# In[10]:


df.Chloramines=df.Chloramines.fillna(0)


# In[11]:


df.Sulfate=df.Sulfate.fillna(0)
df.Conductivity=df.Conductivity.fillna(0)
df.Organic_carbon=df.Organic_carbon.fillna(0)
df.Trihalomethanes=df.Trihalomethanes.fillna(0)
df.Turbidity=df.Turbidity.fillna(0)
df.Potability=df.Potability.fillna(0)


# In[12]:


df


# In[13]:


df.isnull()


# In[14]:


print(df["ph"].mean())#taking the mean value of each column to fill in the place of o


# In[15]:


print(df["Hardness"].mean())
print(df["Solids"].mean())
print(df["Chloramines"].mean())
print(df["Sulfate"].mean())
print(df["Conductivity"].mean())
print(df["Organic_carbon"].mean())
print(df["Trihalomethanes"].mean())
print(df["Turbidity"].mean())


# In[16]:


#converting to ceil values as it will be easy to replace the values
print(math.ceil(df["ph"].mean()))
print(math.ceil(df["Hardness"].mean()))
print(math.ceil(df["Solids"].mean()))
print(math.ceil(df["Chloramines"].mean()))
print(math.ceil(df["Sulfate"].mean()))
print(math.ceil(df["Conductivity"].mean()))
print(math.ceil(df["Organic_carbon"].mean()))
print(math.ceil(df["Trihalomethanes"].mean()))
print(math.ceil(df["Turbidity"].mean()))


# In[17]:


cols=['ph','Hardness','Solids','Chloramines','Sulfate','Conductivity','Organic_carbon','Trihalomethanes','Turbidity']
for col in cols:
    #print(col)
    df.loc[df[col]==0,col]=math.ceil(df[col].mean())


# In[18]:


df


# In[19]:


df.head()


# In[20]:


df.index


# In[21]:


no_of_items=len(df.index)
print(no_of_items)


# In[22]:


item_index_range=[*range(0,3276,1)]


# In[23]:


plt.scatter(x=item_index_range, y=df['ph'],color = 'red',s=9)
plt.show()


# In[24]:


plt.scatter(x=item_index_range,y=df['Hardness'],color='blue',s=9)
plt.show()


# In[25]:


plt.scatter(x=item_index_range,y=df['Solids'],color='purple',s=9)
plt.show()


# In[26]:


plt.scatter(x=item_index_range,y=df['Chloramines'],color='green',s=9)
plt.show()


# In[27]:


plt.scatter(x=item_index_range,y=df['Sulfate'],color='pink',s=9)
plt.show()


# In[28]:


plt.scatter(x=item_index_range,y=df['Conductivity'],color='yellow',s=9)
plt.show()


# In[29]:


plt.scatter(x=item_index_range,y=df['Organic_carbon'],color='cyan',s=9)
plt.show()


# In[30]:


plt.scatter(x=item_index_range,y=df['Trihalomethanes'],color='grey',s=9)
plt.show()


# In[31]:


plt.scatter(x=item_index_range,y=df['Turbidity'],color='brown',s=9)
plt.show()


# In[32]:


df['Solids'].max()


# In[33]:


df['Solids'].min()


# In[34]:


Solids=[]
for i in df['Solids']:
    Solids.append(i)
    


# In[35]:


Solids


# In[36]:


#normalisation n  
for i in range(len(Solids)):
    Solids[i]=((Solids[i]-320.942611274359)/(61227.19600771213-320.942611274359))*(1-0)+0


# In[37]:


Solids


# In[38]:


plt.scatter(x=item_index_range,y=df['Solids'],color='purple',s=9)


# In[39]:


#encoding is not needed in case of our data set as it has no string values


# In[40]:


#Multi layer perceptron classifier (mlp)
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split


# In[41]:


print(df.shape)


# In[42]:


df.describe()


# In[43]:


target_column=['Potability']
unused_column=['Slno']
predictors=list(set(list(df.columns))-set(target_column))
predictors1=list(set(predictors)-set(unused_column))
MaxPred=df[predictors1].max()
df[predictors1]=df[predictors1]/df[predictors1].max()
df.head()


# In[44]:


print(predictors1)


# In[45]:


X=df[predictors1].values
y=df[target_column].values

X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.20)
print(X_train.shape)
print(X_test.shape)


# In[46]:


print(X)


# In[47]:


mlp=MLPClassifier(hidden_layer_sizes=(9,9,9),activation='relu',solver='adam',max_iter=100)
mlp.fit(X_train,y_train)

predict_train=mlp.predict(X_train)
predict_test=mlp.predict(X_test)


# In[48]:


from sklearn.metrics import classification_report,confusion_matrix
print(confusion_matrix(y_train,predict_train))\
print(classification_report(y_train,predict_train))


# In[49]:


#new values for prediction
NewX=[]
print("Enter new ph ")
NewX.append(float(input("Enter valve : ")))
print("enter new Hardness ")
NewX.append(float(input("Enter value : ")))
print("enter new Solids ")
NewX.append(float(input("Enter value : ")))
print("enter new Chloramines ")
NewX.append(float(input("Enter value : ")))
print("enter new Sulfate ")
NewX.append(float(input("Enter value : ")))
print("enter new Conductivity ")
NewX.append(float(input("Enter value : ")))
print("enter new Organic_carbon ")
NewX.append(float(input("Enter value : ")))
print("enter new Trihalomethanes ")
NewX.append(float(input("Enter value : ")))
print("enter new Turbidity ")
NewX.append(float(input("Enter value : ")))


# In[50]:


NewX=np.array(NewX)
NewX.shape


# In[51]:


NewX


# In[52]:


NewX=pd.DataFrame(NewX).transpose()


# In[53]:


NewX.rename(columns = {0:'ph'},inplace=True)
NewX.rename(columns= {1:'Hardness'},inplace=True)
NewX.rename(columns= {2:'Solids'},inplace=True)
NewX.rename(columns= {3:'Chloramines'},inplace=True)
NewX.rename(columns= {4:'Sulfate'},inplace=True)
NewX.rename(columns= {5:'Conductivity'},inplace=True)
NewX.rename(columns= {6:'Organic_carbon'},inplace=True)
NewX.rename(columns= {7:'Trihalomethanes'},inplace=True)
NewX.rename(columns= {8:'Turbidity'},inplace=True)


# In[54]:


NewX


# In[55]:


New=NewX
MaxPred


# In[56]:


NewX[predictors1]=NewX[predictors1]/MaxPred


# In[57]:


New


# In[58]:


NewSample=New[predictors1].values


# In[59]:


NewSample


# In[60]:


NewSample_Result=mlp.predict(NewSample)


# In[61]:


NewSample_Result


# In[67]:


df['Potability'].unique()

