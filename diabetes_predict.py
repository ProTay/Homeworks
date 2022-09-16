import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV
from sklearn import metrics
import numpy as np
def app(df):
	@st.cache()
	def d_tree_pred(df, glucose, bp, insulin, bmi, pedigree, age):
		feature_cols = list(df.columns)
		feature_cols.remove('SkinThickness')
		feature_cols.remove('Pregnancies')
		feature_cols.remove('Outcome')

		x = df[feature_cols]
		y = df['Outcome']
		x_train, x_test, y_train, y_test = train_test_split(x,y,test_size = 0.30, random_state =1)
		dtree_clf = DecisionTreeClassifier(criterion = 'entropy', max_depth =3)
		dtree_clf.fit(x_train, y_train)
		prediction = dtree_clf.predict([[glucose,bp,insulin,bmi,pedigree,age]])
		y_train_pred = dtree_clf.predict(x_train)
		y_test_pred = dtree_clf.predict(x_test)
		prediction = prediction[0]
		score = round(metrics.accuracy_score(y_train, y_train_pred))
		return prediction, score
	def grid_tree_pred(df, glucose, bp, insulin, bmi, pedigree, age):
		feature_cols2 = list(df.columns)
		feature_cols2.remove('SkinThickness')
		feature_cols2.remove('Pregnancies')
		feature_cols2.remove('Outcome')

		x = df[feature_cols2]
		y = df['Outcome']
		x_train, x_test, y_train, y_test = train_test_split(x,y,test_size = 0.30, random_state =1)
		param_grid = {'criterion':['gini','entropy'], 'max_depth':np.arange(4,21), 'random_state' : [42]}
		grid_tree = GridSearchCV(DecisionTreeClassifier(),param_grid, scoring = 'roc_auc', n_jobs = -1)
		grid_tree.fit(x_train, y_train)
		best_tree = grid_tree.best_estimator_
		prediction = best_tree.predict([[glucose,bp,insulin,bmi,pedigree,age]])
		prediction = prediction[0]
		y_train_pred = best_tree.predict(x_train)
		y_test_pred = best_tree.predict(x_test)
		score = round(metrics.accuracy_score(y_train, y_train_pred))
		return prediction, score
	st.markdown("<p style='color:red;font-size:25px'>This app uses <b>Decision Tree Classifier</b> for the Early Prediction of Diabetes.", unsafe_allow_html = True) 
	st.subheader('Select Values:')
	gl = st.slider("Glucose",float(df['Glucose'].min()),float(df['Glucose'].max()))
	bp = st.slider("Blood Pressure",float(df['BloodPressure'].min()),float(df['BloodPressure'].max()))
	insulin = st.slider("Insulin",float(df['Insulin'].min()),float(df['Insulin'].max()))
	bmi = st.slider("BMI",float(df['BMI'].min()),float(df['BMI'].max()))
	pedigree = st.slider("Pedigree Function",float(df['DiabetesPedigreeFunction'].min()),float(df['DiabetesPedigreeFunction'].max()))
	age = st.slider("Age",float(df['Age'].min()),float(df['Age'].max()))
	st.subheader('Model Selection')
	classifier_picker = st.selectbox("Select the Decision Tree Classifier", ('Decision Tree Classifier', 'GridSearchCV Best Tree Classifier'))
	if classifier_picker == 'Decision Tree Classifier':
		if st.button("Predict"):
			prediction, score = d_tree_pred(df, gl, bp, insulin, bmi, pedigree, age)
			if prediction == 1:
				st.info('The person has diabetes or prone to get diabetes')
			else:
				st.info("The person is free diabetes")
			st.write('The best score of this model is', score, '%')
	else:
		if st.button("Predict"):
			prediction, score = grid_tree_pred(df, gl, bp, insulin, bmi, pedigree, age)
			if prediction == 1:
				st.info("The person has diabetes or prone to get diabetes")
			else:
				st.info("The person is free diabetes")
			st.write('The best score of this is', score, "%")