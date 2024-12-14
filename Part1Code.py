import numpy as np
import pandas as pd
import scipy
import statsmodels.api as sm
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn
sns.set()

#Load Data
filepath=r'C:\Users\Wan Qi\Desktop\2024\05 l Data Science Projects\MY Property\NAPIC Selected Data.csv'
df=pd.read_csv(filepath)

#Clean Data
#Clean up column names (removing any leading/trailing spaces)
df.columns=df.columns.str.strip()

#Transform into Date
df['Transaction Date']=pd.to_datetime(df['Month, Year of Transaction Date'],format='%b-%y')

#Remove RM and , in Transaction Price
df['Transaction Price']=df['Transaction Price'].str.replace('RM','',regex=False)

# Remove "RM", commas, and whitespace from the Transaction Price column
df['Transaction Price'] = df['Transaction Price'].str.replace('RM', '', regex=False)
df['Transaction Price']=df['Transaction Price'].str.replace(',','',regex=False)
df['Transaction Price']=df['Transaction Price'].str.strip().astype(float)

#Replace - with 0
df['Main Floor Area']=df['Main Floor Area'].str.replace('-','',regex=False)
df['Unit']=df['Unit'].str.replace('-','',regex=False)

#Drop unncessary columns
df.drop(columns='Month, Year of Transaction Date',inplace=True)
df.drop(columns='Land/Parcel Area',inplace=True)
df.drop(columns='Unit Metrics',inplace=True)
df.drop(columns='Main Floor Area',inplace=True)
df.drop(columns='Unit',inplace=True)

#Change Unit Level into numeric
df['Unit Level'].str.strip()
df['Unit Level']=df['Unit Level'].str.replace('A','',regex=False)
df['Unit Level']=df['Unit Level'].str.replace('UG','1',regex=False)
df['Unit Level']=df['Unit Level'].str.replace('P','100',regex=False)

# Replace both empty strings and NaN values with 0
df['Unit Level'] = df['Unit Level'].replace('', 0).fillna(0)

# Convert 'Unit Level' to numeric after replacements
df['Unit Level'] = pd.to_numeric(df['Unit Level'], errors='coerce')

#Identify PSF
df['PSF']=df['Transaction Price']/df['Square Feet']

#Extract Month & Date
df['Year']=pd.to_datetime(df['Transaction Date']).dt.year
df['Month']=pd.to_datetime(df['Transaction Date']).dt.month

#Analysing Indera Subang
InderaSubang_data=df[df['Scheme Name/Area']=="INDERA SUBANG KONDOMINIUM - USJ 6"]

df.sort_values(by='Transaction Date',ascending=True, inplace=True)
