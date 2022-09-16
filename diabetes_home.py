import pandas as pd
import streamlit as st
import diabetes_home
import diabetes_predict
import diabetes_plot
data_frame = pd.read_csv('Diabetes.csv')

st.sidebar.title('Navigation')
navigation = {"Home":diabetes_home, "Predict Diabetes": diabetes_predict, "Visualise Decision Tree": diabetes_plot}
page = st.sidebar.radio('Go to', tuple(navigation.keys()))
if page == "Home":
	st.title('Early Diabetes Predition Web App')
	st.write("Diabetes is a chronic (long-lasting) health condition that afects how your body turns food into energy. There isn't a cure yet for diabetes, but losing weight, eating healthy food, and being active can really help in reducing the impact of diabetes. This Web app will help you to predict whether a person has diabetes or is prone to get bdiabetes in future by analysing the values of several features using the Decision Treee Classifier.")
	st.subheader('View Data')
	with st.beta_expander('View Data'):
		st.dataframe(data_frame)
	st.subheader('Column Description:')
	beta_01, beta_02, beta_03 = st.beta_columns(3)
	with beta_01:
		if st.checkbox('Show all columns names'):
			st.table(data_frame.columns)
	with beta_02:
		if st.checkbox('View columns data-types'):
			st.table(data_frame.dtypes)
	with beta_03:
		if st.checkbox('View column data'):
			select = st.selectbox("Select Columns",data_frame.columns)
			st.table(data_frame[select])
	if st.checkbox('Show summary'):
		st.table(data_frame.describe())
else:
	navigate = navigation[page]
	navigate.app(data_frame)