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
df = pd.read_csv("Depresso.csv")
df.drop(['AGERNG', 'GENDER', 'EDU', 'PROF', 'MARSTS', 'RESDPL', 'LIVWTH',  'DEBT', 'PHYEX', 'SMOKE', 'DRINK',
        'ILLNESS', 'PREMED', 'EATDIS', 'AVGSLP', 'TSSN', 'WRKPRE'], axis=1, inplace=True)

categ = ['ENVSAT', 'POSSAT', 'FINSTR', 'INSOM', 'ANXI',
        'DEPRI', 'ABUSED', 'CHEAT', 'THREAT', 'SUICIDE',
        'INFER', 'CONFLICT', 'LOST', 'DEPRESSED']
# label_encoder object knows how to understand word labels...apparently
le = preprocessing.LabelEncoder()
# Encode Categorical Columns
df[categ] = df[categ].apply(le.fit_transform)
x = df.drop('DEPRESSED', axis = 1)
y = df['DEPRESSED']


class ml:
    def __init__(self, model_type):
        
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=109)

        if model_type == 'decision tree':
            model = DecisionTreeClassifier(criterion = 'entropy', random_state = 0) # decision tree
        elif model_type == 'naive bayes': 
            model = GaussianNB()    # naive-bayes
        elif model_type == 'svm':
            model = SVC(decision_function_shape='ovo')  # SVM
        elif model_type == 'knn':
            model= KNeighborsClassifier(n_neighbors=5, metric='minkowski', p=2 )    # KNN
        elif model_type == 'random forest':
            model = RandomForestRegressor() # random forest
        elif model_type == 'logistic regression':
            model = LogisticRegression()    # logistic regression
        elif model_type == 'dnn':
            model = self.check_for_dnn()
        else:
            raise Exception("Model name parameter is invalid!")
        if model_type != 'dnn':
            model.fit(x_train, y_train)

        self.model_type = model_type
        self.model = model


    def predict(self, arglist):
        if self.model_type == 'dnn':
            y_pred = self.model.predict(arglist)[0]
            y_pred = y_pred.tolist()[0]
            y_pred = round(y_pred, 2)
        else:
            y_pred = self.model.predict(arglist)[0]
        return y_pred


    def check_for_dnn(self):
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
    

class decision_tree:

    model = []

    @staticmethod
    def train():
        from sklearn.tree import DecisionTreeClassifier
        from sklearn.model_selection import train_test_split

        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3,random_state=109)
        model = DecisionTreeClassifier(criterion = 'entropy', random_state = 0)
        model.fit(x_train, y_train)
        decision_tree.model = model
        y_pred = model.predict(x_test)
        return

    @staticmethod
    def predict(arglist):
        model = decision_tree.model
        if(model==[]):
            print("Train decision tree")
            return
        x_pred = arglist #[envsat, possat, finstr, insom, anx, depri, abuse, cheat, threat, suic, infer, conf, loss]
        y_pred = model.predict(x_pred)[0]
        print("Decision tree:", y_pred)
        return (y_pred, -1)

class naivebayes:

    model = []

    @staticmethod
    def train():
        from sklearn.naive_bayes import GaussianNB
        from sklearn.model_selection import train_test_split


        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3,random_state=109)
        model = GaussianNB()
        model.fit(x_train, y_train)
        y_pred = model.predict(x_test)
        naivebayes.model = model
        return

    @staticmethod
    def predict(arglist):
        model = naivebayes.model
        if(model==[]):
            print("Train naive bayes")
            return
        x_pred = arglist
        y_pred = model.predict(x_pred)[0]
        print("Naive Bayes:", y_pred)
        return (y_pred, -1)

class svm:

    model = []
    
    @staticmethod
    def train():
        from sklearn.svm import SVC
        from sklearn.model_selection import train_test_split

        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3,random_state=109)
        model = SVC(decision_function_shape='ovo')
        model.fit(x_train, y_train)
        svm.model = model
        y_pred = model.predict(x_test)
        return

    @staticmethod
    def predict(arglist):
        model = svm.model
        if(model==[]):
            print("Train svm")
            return
        x_pred = arglist
        y_pred = model.predict(x_pred)[0]
        print("SVM:" , y_pred)
        return (y_pred, -1)

class knn:

    model = []
    
    @staticmethod
    def train():
        from sklearn.neighbors import KNeighborsClassifier
        from sklearn.model_selection import train_test_split

        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3,random_state=109)
        model= KNeighborsClassifier(n_neighbors=5, metric='minkowski', p=2 )  
        model.fit(x_train, y_train) 
        y_pred = model.predict(x_test)
        knn.model = model
        return

    @staticmethod
    def predict(arglist):
        model = knn.model
        if(model==[]):
            print("Train knn")
            return
        x_pred = arglist
        y_pred = model.predict(x_pred)[0]
        print("KNN:", y_pred)
        return (y_pred, -1)

class random_forest:

    model = []

    @staticmethod
    def train():
        from sklearn.ensemble import RandomForestRegressor
        from sklearn.model_selection import train_test_split


        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3,random_state=109)
        model = RandomForestRegressor()
        model.fit(x_train, y_train) 
        random_forest.model = model
        return

    @staticmethod
    def predict(arglist):
        model = random_forest.model
        if(model==[]):
            print("Train random forest")
            return
        x_pred = arglist
        y_pred = model.predict(x_pred)[0]
        y_pred_rounded = round(y_pred)
        print("Random forest:", y_pred)
        print("Random forest rounded:",y_pred_rounded)
        return (y_pred_rounded, y_pred)

class logistic_regression:

    model = []

    @staticmethod
    def train():
        from sklearn.linear_model import LogisticRegression
        from sklearn.model_selection import train_test_split

        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3,random_state=109)
        model = LogisticRegression()
        model.fit(x_train, y_train) 
        y_pred = model.predict(x_test)
        logistic_regression.model = model
        return

    @staticmethod
    def predict(arglist):
        model = logistic_regression.model
        if(model==[]):
            print("Train logistic regression")
            return
        x_pred = arglist
        y_pred = model.predict(x_pred)[0]
        print("Logistic regression:", y_pred)
        return (y_pred, -1)

class dlseq:

    model = []

    @staticmethod
    def loadmodel():
        from keras.models import load_model
        dlseq.model = load_model('seq_model')
        print("Model loaded")
        return

    @staticmethod
    def train():
        from keras.models import Sequential
        from keras.layers import Dense
        from sklearn.model_selection import train_test_split

        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3,random_state=109)
        model = Sequential()
        model.add(Dense(50, activation='relu'))
        model.add(Dense(20, activation='relu'))
        model.add(Dense(1, activation='sigmoid'))
        model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
        model.fit(x_train, y_train, epochs=20)
        model.save('seq_model')
        dlseq.model = model
        print("Model saved")
        print("DL Sequential accuracy:", model.evaluate(x_test,y_test,verbose=1))
        return

    @staticmethod
    def check_for_model():
        try:
            dlseq.loadmodel()
        except:
            dlseq.train()
        return
    
    @staticmethod
    def predict(arglist):
        model = dlseq.model
        if(model==[]):
            print("DNN model unavailable")
            return
        else:
            print('Using DNN model')
        x_pred = arglist
        y_pred = model.predict(x_pred)[0]
        y_pred_rounded = round(float(y_pred))
        print("Deep learning sequential original:", y_pred)
        print("Deep learning sequential rounded:", y_pred_rounded)
        return (y_pred_rounded, y_pred[0])


if __name__ == '__main__':
    from server import request_handler
    values = ["0","0","0","0","0","0","0","0","0","0","0","0","0"]
    model_num = "5"
    pred = request_handler(values, model_num)
    print(pred)
    