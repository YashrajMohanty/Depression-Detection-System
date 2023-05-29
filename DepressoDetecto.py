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


class ml:
    def __init__(self):
        
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=109)

        self.dec_tree = DecisionTreeClassifier(criterion = 'entropy', random_state = 0) # decision tree
        self.nb = GaussianNB()    # naive-bayes
        self.svm = SVC(decision_function_shape='ovo')  # SVM
        self.knn= KNeighborsClassifier(n_neighbors=5, metric='minkowski', p=2 )    # KNN
        self.random_forest = RandomForestRegressor() # random forest
        self.logistic_reg = LogisticRegression()    # logistic regression
        self.dnn = self._check_for_dnn()
        
        for model in [self.dec_tree, self.nb, self.svm, self.knn, self.random_forest, self.logistic_reg]:
            model.fit(x_train, y_train)


    def predict(self, arglist, model_type):
        if model_type == 'decision tree':
            model = self.dec_tree
        elif model_type == 'naive bayes': 
            model = self.nb
        elif model_type == 'svm':
            model = self.svm
        elif model_type == 'knn':
            model = self.knn
        elif model_type == 'random forest':
            model = self.random_forest
        elif model_type == 'logistic regression':
            model = self.logistic_reg
        elif model_type == 'dnn':
            model = self.dnn
        else:
            raise Exception("model_type parameter invalid!")

        if model_type == 'dnn':
            y_pred = model.predict(arglist)[0]
            y_pred = y_pred.tolist()[0]
            y_pred = round(y_pred, 2)
        else:
            y_pred = model.predict(arglist)[0]
        return y_pred


    def _check_for_dnn(self):
        try:
            model = self._load_dnn()
        except:
            model = self._train_dnn()
        return model


    def _load_dnn(self):
        from keras.models import load_model
        model = load_model('seq_model')
        print("DNN loaded")
        return model


    def _train_dnn(self):

        x_train, _, y_train, _ = train_test_split(x, y, test_size=0.3, random_state=109)
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
    

if __name__ == '__main__':
    from server import request_handler
    values = ["0","0","0","0","0","0","0","0","0","0","0","0","0"]
    model_num = "5"
    pred = request_handler(values, model_num)
    print(pred)