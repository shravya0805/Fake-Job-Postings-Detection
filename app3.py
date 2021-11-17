import streamlit as st 
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import plot_confusion_matrix, plot_roc_curve, plot_precision_recall_curve
import sklearn.metrics as mt
import time

def app():
    st.markdown("<h1 style='text-align: center; color: #f6c453';>Fake Job Listing</h1>", unsafe_allow_html=True)      
    
    dataset_name = 'fake_job_postings'


    st.sidebar.header("Select classifier")
    classifier_name = st.sidebar.selectbox('classifier',('SVM','Naive Bayes','Logistic Regression',  'KNN','Decision Tree','Neural Network')
    )


    
    def get_dataset(name):
        data = None
        fake_job_postings = pd.read_csv(r'C:\Users\shrav\Downloads\fake_job_postings_cleaned (2).csv')
      
        X = fake_job_postings[['text']]
        y = fake_job_postings['fraudulent']
        return X, y

    X, y = get_dataset(dataset_name)

    def add_parameter_ui(clf_name):
        time.sleep(2)
        params = dict()
        if clf_name == 'SVM':
            st.sidebar.subheader("Hyperparameters")
            C = st.sidebar.number_input("C", 0.01, 10.0, step=0.01, key="C")
            params['C'] = C 
            gamma=st.sidebar.radio('Gamma',['auto', 'scale'])
            params['gamma']=gamma
            kernel=st.sidebar.selectbox('Kernel',('rbf', 'sigmoid','linear','poly'))
            params['kernel'] = kernel
            if kernel == 'poly':
                degree=st.sidebar.slider('Degree',1,5)
                params['degree'] = degree
            else:
                params['degree'] = 3  
            max_iter = st.sidebar.slider("Maximum iterations", 100, 500, key="max_iter")
            params['max_iter'] = max_iter
            
        elif clf_name == 'KNN':
            st.sidebar.subheader("Hyperparameters")
            K = st.sidebar.slider('K', 1, 15)
            params['K'] = K
            weights=st.sidebar.selectbox('Weights',('uniform', 'distance'))
            params['weights'] = weights
            algorithm=st.sidebar.radio('Algorithm',['auto', 'ball_tree','kd_tree','brute'])
            params['algorithm'] = algorithm
            
        elif clf_name == 'Neural Network':
            st.sidebar.subheader("Hyperparameters")
            learning_rate_init =st.sidebar.number_input("learning_rate_init", 0.01, 1.0, step=0.01, key="learning_rate_init")
            params['learning_rate_init'] = learning_rate_init
            activation = st.sidebar.radio('Activation',['relu', 'logistic','tanh','identity'])
            params['activation'] = activation
            solver = st.sidebar.selectbox('Solver',('adam', 'lbfgs','sgd'))
            params['solver'] = solver
            max_iter = st.sidebar.slider("Maximum iterations", 100, 500, key="max_iter")
            params['max_iter'] = max_iter  
        elif clf_name == 'Decision Tree':
            st.sidebar.subheader("Hyperparameters")
            max_depth = st.sidebar.slider('max_depth', 1, 15)
            params['max_depth'] = max_depth
            criterion = st.sidebar.radio('Criterion',['gini', 'entropy'])
            params['criterion'] = criterion
            splitter = st.sidebar.selectbox('Splitter',('best', 'random'))
            params['splitter'] = splitter    
        elif clf_name == 'Naive Bayes':
            st.sidebar.subheader("Hyperparameters")
            alpha=st.sidebar.number_input("Alpha", 0.01, 1.0, step=0.01, key="Alpha")
            params['alpha'] = alpha
            fit_prior=st.sidebar.selectbox('Fit_prior',(True, False))
            params['fit_prior'] = fit_prior   
        else:
            st.sidebar.subheader("Hyperparameters")
            C = st.sidebar.number_input("C (Regularization parameter)", 0.01, 10.0, step=0.01, key="C")
            params['C'] = C   
            solver = st.sidebar.selectbox('Solver',('lbfgs', 'newton-cg','saga','sag'))
            params['solver'] = solver
            multi_class=st.sidebar.radio('Multi_class',['auto', 'multinomial','ovr'])
            params['multi_class'] = multi_class
            max_iter = st.sidebar.slider("Maximum iterations", 100, 500, key="max_iter")
            params['max_iter'] = max_iter    
        return params
    
    params = add_parameter_ui(classifier_name)

    def get_classifier(clf_name, params):
        clf = None
        if clf_name == 'SVM':
            clf = SVC(C=params['C'],kernel=params['kernel'],degree=params['degree'],gamma=params['gamma'],max_iter=params['max_iter'])
        elif clf_name == 'KNN':
            clf = KNeighborsClassifier(n_neighbors=params['K'],weights=params['weights'],algorithm=params['algorithm'])
        elif clf_name == 'Naive Bayes':
            clf=MultinomialNB(alpha=params['alpha'],fit_prior=params['fit_prior'])
        elif clf_name == 'Logistic Regression':
            clf = LogisticRegression(C=params['C'],solver=params['solver'],multi_class=params['multi_class'],max_iter=params['max_iter'])
        elif clf_name == 'Neural Network':   clf=MLPClassifier(solver=params['solver'],random_state=1234,learning_rate_init=params['learning_rate_init'],activation=params['activation'],max_iter=params['max_iter'])
        else:
            clf = DecisionTreeClassifier(max_depth=params['max_depth'],splitter=params['splitter'],criterion=params['criterion'])
        return clf

    clf = get_classifier(classifier_name, params)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=53)
    count_vectorizer = CountVectorizer(stop_words='english')
    X_train = count_vectorizer.fit_transform(X_train.text.values)
    X_test = count_vectorizer.transform(X_test.text.values)
 
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)

    acc = accuracy_score(y_test, y_pred)

    st.subheader(f'{classifier_name} Results')
    st.text('Model Report:\n ' + mt.classification_report(y_test, y_pred))

   
    def color(classifier_name):
        if classifier_name == 'Logistic Regression':
            return "Blues"
        elif classifier_name =='Naive Bayes':
            return "pink"
        elif classifier_name == 'SVM':
            return "inferno"
        elif classifier_name == 'Neural Network':
            return "cividis"
        elif classifier_name == 'KNN':
            return "Wistia_r"
        else:
            return "Spectral"
    
    st.set_option('deprecation.showPyplotGlobalUse', False)   
    
    
    def plot_metrics(metrics_list):
        if "Confusion Matrix" in metrics_list: 
            st.subheader("Confusion Matrix")
            plot_confusion_matrix(clf, X_test, y_test,cmap=color(classifier_name))
            st.pyplot()
        if "ROC Curve" in metrics_list:
            st.subheader("roc_curve")  
            plot_roc_curve(clf, X_test, y_test)
            st.pyplot()
        if "Precision-Recall Curve" in metrics_list:
            st.subheader("precision_recall_curve")
            plot_precision_recall_curve(clf, X_test, y_test)
            st.pyplot()
    metrics = st.sidebar.multiselect("What metrics to plot?", ("Confusion Matrix", "ROC Curve", "Precision-Recall Curve"))
    plot_metrics(metrics)

    