import numpy as numpy
import pandas as pd
import streamlit as st

def app(census_df):
	with st.beta_expander('View Data'):
		st.dataframe(census_df)
	if st.checkbox('Display summary'):
		st.table(census_df.describe())