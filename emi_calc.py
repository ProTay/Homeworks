import streamlit as st
def calculate_emi(p,n,r):
	emi_1 = p * (r/100) 
	emi_2 = ((1 + (r/100))** n)
	emi_3 = emi_2 /( emi_2 - 1)
	emi = emi_1 * emi_3
	return emi

st.title('EMI Calculator')

p = st.sidebar.slider('Input Principal', 1000,1000000)
n = st.sidebar.slider('Input Tenure', 1,30)
r = st.sidebar.slider('Input Roi', 1, 15)

emi = st.write('Calculation:',calculate_emi(p,n,r))

m = st.sidebar.slider('Input OBA', (1-(n*12)), 0)

def improvised_emi(p,n,r,m):
	emi_1 = p * (1+r/100)**n 
	emi_2 = ((1+r/100)**m)
	emi_3 = emi_1 - emi_2
	emi_4 = ((1 + r/100)** n) - 1
	emi = emi_3/emi_4
	return emi
emi_2 = st.write('Imp Calculation:',improvised_emi(p,n,r,m))

if st.sidebar.button('Calculate EMI'):
	emi
if st.sidebar.button('Calculate Improvised EMI'):
	emi_2