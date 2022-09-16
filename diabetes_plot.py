from sklearn import metrics
from sklearn.metrics import classification_report, plot_confusion_matrix, confusion_matrix
import warnings
from sklearn.tree import export_graphviz
from io import StringIO
import streamlit as st  
import pandas as pd
from sklearn.model_selection import GridSearchCV
from sklearn import tree
from sklearn.tree import DecisionTreeClassifier
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
def app(df):
	st.set_option('deprecation.showPyplotGlobalUse', False)
	st.title('Visualise the Diabetes Predition Web app')
	if st.checkbox('View correlation heatmap'):
		st.subheader("Correlation Heatmap")
		plt.figure(figsize = (10, 6))
		ax = sns.heatmap(df.iloc[:,1:].corr(), annot =True)
		bottom, top = ax.get_ylim()
		ax.set_ylim(bottom + 0.5, top - 0.5)
		st.pyplot()
	model_select = st.selectbox('Select Model',('Decision Tree Classifier', 'GridSearchCV Best Tree Classifier'))
	feature_cols = list(df.columns)
	feature_cols.remove('Pregnancies')
	feature_cols.remove('SkinThickness')
	feature_cols.remove('Outcome')

	x = df[feature_cols]
	y = df['Outcome']
	x_train, x_test, y_train, y_test = train_test_split(x,y,test_size = 0.30, random_state = 1)
	dtree = DecisionTreeClassifier(max_depth =3, criterion = 'entropy')
	dtree.fit(x_train, y_train)
	y_train_pred = dtree.predict(x_train)
	y_test_pred = dtree.predict(x_test)
	if model_select == 'Decision Tree Classifier':
		if st.checkbox('Plot confusion_matrix'):
			plt.figure(figsize = (10,6))
			plot_confusion_matrix(dtree, x_train, y_train, values_format = 'd')
			st.pyplot()
		if st.checkbox('Plot Decision Tree'):
			dot_data = tree.export_graphviz(decision_tree = dtree, max_depth = 3, out_file = None, filled = True, rounded = True,feature_names = feature_cols, class_names = ['0','1'])
			st.graphviz_chart(dot_data)
	if model_select == 'GridSearchCV Best Tree Classifier':
		param_grid = {'criterion':['gini','entropy'], 'max_depth':np.arange(4,21),'random_state' : [42]}
		grid_tree = GridSearchCV(DecisionTreeClassifier(), param_grid, scoring = 'roc_auc', n_jobs = -1)
		grid_tree.fit(x_train,y_train)
		best_tree = grid_tree.best_estimator_
		y_train_pred2 = grid_tree.predict(x_train)
		y_test_pred2 = grid_tree.predict(x_test)
		if st.checkbox('Plot confusion_matrix'):
			plt.figure(figsize = (10,6))
			plot_confusion_matrix(grid_tree, x_train, y_train, values_format = 'd')
			st.pyplot()
		if st.checkbox('Plot Decision Tree'):
			dot_data = tree.export_graphviz(decision_tree = best_tree, max_depth = 3, out_file = None, filled = True, rounded= True,feature_names = feature_cols,class_names =['0','1'] )
			st.graphviz_chart(dot_data)


