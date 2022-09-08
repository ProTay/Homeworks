import pandas as pd
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns

st.set_option('deprecation.showPyplotGlobalUse', False)


def app(census_df):
	st.title('Visualisation data')
	plot_list = st.sidebar.multiselect("Select Plot Types:",('Pie Chart', 'Box Chart', 'Count Plot'))
	
	if 'Pie Chart' in plot_list:
		gender_data = census_df['gender'].value_counts()
		income_data = census_df['income'].value_counts()
		pie_data = [gender_data,income_data]
		st.subheader('Plot Charts')
		
		for i in pie_data:
			plt.figure(figsize = (13,6))
			plt.pie(i, labels = i.index, autopct = '%1.2f%%', explode = np.linspace(0.1,0.2,2))
			st.pyplot()
	if 'Box Chart' in plot_list:
		income_bdata = 'income'
		gender_bdata = 'gender'
		box_data = [income_bdata, gender_bdata]
		for i in box_data:
			plt.figure(figsize = (12,2))
			st.subheader(f'Box Plot for {i}')
			sns.boxplot(x = 'hours-per-week' , y = i, data = census_df)
			st.pyplot()
	if 'Count Plot' in plot_list:
		st.subheader('Count Plot for workclass')
		plt.figure(figsize = (12,6))
		sns.countplot(x = 'workclass', data = census_df)
		st.pyplot()
