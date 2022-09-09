# Open Sublime text editor, create a new Python file, copy the following code in it and save it as 'census_main.py'.

# Import modules
import numpy as np
import pandas as pd
import streamlit as st
st.set_page_config(page_title = 'Census Visualisation Web App',page_icon = ':random:',layout = 'centered',initial_sidebar_state = 'auto')

@st.cache()
def load_data():
	# Load the Adult Income dataset into DataFrame.

	df = pd.read_csv('adult.csv', header=None)
	df.head()

	# Rename the column names in the DataFrame using the list given above. 

	# Create the list
	column_name =['age', 'workclass', 'fnlwgt', 'education', 'education-years', 'marital-status', 'occupation', 
               'relationship', 'race','gender','capital-gain', 'capital-loss', 'hours-per-week', 'native-country', 'income']

	# Rename the columns using 'rename()'
	for i in range(df.shape[1]):
	  df.rename(columns={i:column_name[i]},inplace=True)

	# Print the first five rows of the DataFrame
	df.head()

	# Replace the invalid values ' ?' with 'np.nan'.

	df['native-country'] = df['native-country'].replace(' ?',np.nan)
	df['workclass'] = df['workclass'].replace(' ?',np.nan)
	df['occupation'] = df['occupation'].replace(' ?',np.nan)

	# Delete the rows with invalid values and the column not required 

	# Delete the rows with the 'dropna()' function
	df.dropna(inplace=True)

	# Delete the column with the 'drop()' function
	df.drop(columns='fnlwgt',axis=1,inplace=True)

	return df

census_df = load_data()
st.title('Census Visualisation Web App')
st.subheader('This web app allows a user to explore and visualise the census data')
st.title('View Data')
with st.beta_expander('View Dataset'):
	st.dataframe(census_df)
st.title('Columns Description:')
beta_01, beta_02, beta_03 = st.beta_columns(3)
with beta_01:
	if st.checkbox('Show all column names'):
		st.table(list(census_df.columns))
with beta_02:
	if st.checkbox('View Column data-type'):
		st.table(census_df.dtypes)
with beta_03:
	if st.checkbox('View Column Data'):
		column_data = st.selectbox('Select Column', tuple(census_df.columns))
		st.write(census_df[column_data])
if st.checkbox('Show summary'):
	st.table(census_df.describe())

