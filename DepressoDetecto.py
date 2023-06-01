import pandas as pd
from sklearn import preprocessing
import warnings
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LogisticRegression
from keras.models import Sequential
from keras.layers import Dense

warnings.filterwarnings("ignore", category=UserWarning) #suppress UserWarning

'''Data cleaning'''
df = pd.read_csv("DepressoFinal.csv")
categ = list(df.columns)

# label_encoder object knows how to understand word labels...apparently
le = preprocessing.LabelEncoder()
# Encode Categorical Columns
df[categ] = df[categ].apply(le.fit_transform)
x = df.drop('DEPRESSED', axis = 1)
y = df['DEPRESSED']

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=109)
dnn_force_train = False # forces model to be trained even if its already available

def _check_for_dnn():
    try:
        model = _load_dnn()
    except:
        model = _train_dnn()
    return model

def _load_dnn():
    if dnn_force_train:
        raise Exception('dnn_force_train set to True')
    from keras.models import load_model
    model = load_model('seq_model')
    print("DNN loaded")
    return model

def _train_dnn(x_train=x_train, y_train=y_train):

    model = Sequential()
    model.add(Dense(50, activation='relu'))
    model.add(Dense(20, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    model.fit(x_train, y_train, epochs=20)
    model.save('seq_model')
    print("Saved DNN")
    print('DNN loaded')
    return model

dec_tree = DecisionTreeClassifier(criterion = 'entropy', random_state = 0) # decision tree
nb = GaussianNB()    # naive-bayes
svm = SVC(decision_function_shape='ovo')  # SVM
knn= KNeighborsClassifier(n_neighbors=5, metric='minkowski', p=2 )    # KNN
random_forest = RandomForestRegressor() # random forest
logistic_reg = LogisticRegression()    # logistic regression
dnn = _check_for_dnn()
for model in [dec_tree, nb, svm, knn, random_forest, logistic_reg]:
    model.fit(x_train, y_train)

def predict(arglist, model_type):
    if model_type == 'decision tree':
        model = dec_tree
    elif model_type == 'naive bayes': 
        model = nb
    elif model_type == 'svm':
        model = svm
    elif model_type == 'knn':
        model = knn
    elif model_type == 'random forest':
        model = random_forest
    elif model_type == 'logistic regression':
        model = logistic_reg
    elif model_type == 'dnn':
        model = dnn
    else:
        raise Exception("model_type parameter invalid!")

    if model_type == 'dnn':
        y_pred = model.predict(arglist)[0]
        y_pred = y_pred.tolist()[0]
        y_pred = round(y_pred, 2)
    else:
        y_pred = model.predict(arglist)[0]
    return y_pred


if __name__ == '__main__':
    from server import request_handler
    values = [0,0,0,1,0,1,0,1,1,1,0,0,0]
    model_num = 7
    pred = request_handler(values, model_num)
    print(pred)