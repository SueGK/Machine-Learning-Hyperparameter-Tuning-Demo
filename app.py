import streamlit as st
st.set_page_config(page_title='Machine Learning Hyperparameter Tuning Demo',
                   layout='wide')

import streamlit as st  
import numpy as np 
import pandas as pd 
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score,classification_report,plot_confusion_matrix
import plotly.graph_objects as go 

st.set_option('deprecation.showPyplotGlobalUse', False)
st.write("""
# Machine Learning Hyperparameter Tuning Demo
### **(Heart Disease Prediction Model)**""")
df = pd.read_csv('https://github.com/SueGK/Machine-Learning-Hyperparameter-Tuning-Demo/blob/main/heart.csv')
st.subheader('Dataset')
st.markdown('First 5 lines of dataset')
st.write(df.head(5))


# sidebar
st.sidebar.header('Set hyperparameters for grid search') 
split_size = st.sidebar.slider(
  'The proportion of data (% Training set)', 50, 90, 75, 5)

st.sidebar.subheader('Machine Learning Parameters')
parameter_n_estimators = st.sidebar.slider(
              'Random Forest (n_estimators)',
              0, 500, (10,50), 50)
parameter_n_estimators_step = st.sidebar.number_input('n_estimators stride', 10)

st.sidebar.write('---')
parameter_max_features =st.sidebar.multiselect(
                '(You can choose multiple options)',
                ['auto', 'sqrt', 'log2'],
                ['auto'])

parameter_max_depth = st.sidebar.slider(
            'Maximum depth', 5, 15, (5,8), 2)
parameter_max_depth_step=st.sidebar.number_input(
            'max_depth stride', 1,3)

st.sidebar.write('---')
parameter_criterion = st.sidebar.selectbox('criterion', ('gini', 'entropy'))

st.sidebar.write('---')
parameter_cross_validation=st.sidebar.slider(
          'cross validation', 2, 10)

st.sidebar.subheader('Other parameters')
parameter_random_state = st.sidebar.slider(
          'random seed', 0, 1000, 42, 1)
parameter_bootstrap = st.sidebar.select_slider(
          'bootstrap',
          options=[True, False])
parameter_n_jobs = st.sidebar.select_slider(
          'n_jobs',
          options=[1, -1])


if st.button('Construct'):
  # one hot code
  dataset = pd.get_dummies(df, 
                            columns=['sex', 'cp', 'fbs', 'restecg',
                                    'exang', 'slope', 'ca', 'thal'])
  Y = dataset['target']
  X = dataset.drop(['target'], axis = 1)
  X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=100-split_size)
  sc = StandardScaler()
  X_train = sc.fit_transform(X_train)
  X_test = sc.transform(X_test)



  n_estimators_range = np.arange(parameter_n_estimators[0], parameter_n_estimators[1]+parameter_n_estimators_step, parameter_n_estimators_step)
  """
  if   parameter_n_estimators[0] is 5 
  and  parameter_n_estimators[1] 25 
  and  parameter_n_estimators_step is 5
  then array will be [5,10,15,20,25]
  """
  max_depth_range =np.arange(parameter_max_depth[0],
                            parameter_max_depth[1]+parameter_max_depth_step, 
                            parameter_max_depth_step)
  param_grid = dict(max_features=parameter_max_features,
  n_estimators=n_estimators_range,max_depth=max_depth_range)

  rf = RandomForestClassifier(random_state=parameter_random_state,
                              bootstrap=parameter_bootstrap,
                              n_jobs=parameter_n_jobs)
  grid = GridSearchCV(estimator=rf, param_grid=param_grid, cv=parameter_cross_validation)
  grid.fit(X_train,Y_train)
  st.subheader('Model Performance')
  Y_pred_test = grid.predict(X_test)

  st.write('Model accuracy score')
  # st.info(accuracy_score(Y_test, Y_pred_test))

  st.write("Best parameters are %s，and best score is %0.2f" % (
              grid.best_params_, grid.best_score_))

  st.subheader('Model Parameters')
  st.write(grid.get_params())


  ### ------- 3D viz

  grid_results=pd.concat(
          [pd.DataFrame(grid.cv_results_["params"]),
          pd.DataFrame(grid.cv_results_["mean_test_score"],
          columns=["accuracy"])],
          axis=1)

  grid_contour = grid_results.groupby(['max_depth','n_estimators']).mean()
  grid_reset = grid_contour.reset_index()
  grid_reset.columns = ['max_depth', 'n_estimators', 'accuracy']
  grid_pivot = grid_reset.pivot('max_depth', 'n_estimators')
  x = grid_pivot.columns.levels[1].values
  y = grid_pivot.index.values
  z = grid_pivot.values

  # define layout and axis
  layout = go.Layout(
    xaxis=go.layout.XAxis(
      title=go.layout.xaxis.Title(
        text='n_estimators')
    ),
    yaxis=go.layout.YAxis(
      title=go.layout.yaxis.Title(
        text='max_depth')
    ) )
  fig = go.Figure(data= [go.Surface(z=z, y=y, x=x)], layout=layout )
  fig.update_layout(title='Hyperparameter tuning',
                    scene = dict(
                      xaxis_title='n_estimators',
                      yaxis_title='max_depth',
                      zaxis_title='accuracy'),
                    autosize=False,
                    width=800, height=800,
                    margin=dict(l=65, r=50, b=65, t=90))
  st.plotly_chart(fig)

  st.subheader("Classification Report")

  clf=classification_report(Y_test, Y_pred_test, labels=[0,1],output_dict=True)
  st.write("""
      ### Category 0(Don't have heart disease) :
      Precision : %0.2f     
      Recall : %0.2f      
      F1-score  : %0.2f"""%(clf['0']['precision'],clf['0']['recall'],clf['0']['f1-score']))
  st.write("""
      ### Category 1(Have heart disease) :
      Precision : %0.3f    
      Recall : %0.3f      
      F1-score  : %0.3f"""%(clf['1']['precision'],clf['1']['recall'],clf['1']['f1-score']))
  st.subheader("Confusion Matrix")

  plot_confusion_matrix(grid, X_test, Y_test,display_labels=['Have heart disease',"Don't have heart disease"])

  st.pyplot()
  
