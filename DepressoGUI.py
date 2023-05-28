import pandas as pd
import numpy as np
from sklearn import preprocessing
import warnings

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


'''Config.txt read'''
config = open("Config.txt", "r")
load_dnn_flag = config.readline().split()[2]
load_dnn_flag = load_dnn_flag.lower() in "true"
config.close()


class decision_tree:

    model = []

    @staticmethod
    def train():
        from sklearn.tree import DecisionTreeClassifier
        from sklearn.model_selection import train_test_split
        from sklearn import metrics

        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3,random_state=109)
        model = DecisionTreeClassifier(criterion = 'entropy', random_state = 0)
        model.fit(x_train, y_train)
        decision_tree.model = model
        y_pred = model.predict(x_test)
        print("Decision tree accuracy:", metrics.accuracy_score(y_test, y_pred))
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
        from sklearn import metrics

        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3,random_state=109)
        model = GaussianNB()
        model.fit(x_train, y_train)
        y_pred = model.predict(x_test)
        naivebayes.model = model
        print("Naive Bayes accuracy:", metrics.accuracy_score(y_test, y_pred))
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
        from sklearn import metrics

        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3,random_state=109)
        model = SVC(decision_function_shape='ovo')
        model.fit(x_train, y_train)
        svm.model = model
        y_pred = model.predict(x_test)
        print("SVM accuracy:", metrics.accuracy_score(y_test, y_pred))
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
        from sklearn import metrics

        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3,random_state=109)
        model= KNeighborsClassifier(n_neighbors=5, metric='minkowski', p=2 )  
        model.fit(x_train, y_train) 
        y_pred = model.predict(x_test)
        knn.model = model
        print("KNN accuracy:", metrics.accuracy_score(y_test, y_pred))
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
        #from sklearn import metrics

        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3,random_state=109)
        model = RandomForestRegressor()
        model.fit(x_train, y_train) 
        random_forest.model = model
        # Due to an error, accuracy for random forest is unavailable
        #print("Random forest accuracy:", metrics.accuracy_score(y_test, y_pred))
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
        from sklearn import metrics

        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3,random_state=109)
        model = LogisticRegression()
        model.fit(x_train, y_train) 
        y_pred = model.predict(x_test)
        logistic_regression.model = model
        print("Logistic regression accuracy:", metrics.accuracy_score(y_test, y_pred))
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

class gui:
    
    arglist = None
    algo_option = None

    @staticmethod
    def main():
        startscreen.run()
        homescreen.run()
    
    @staticmethod
    def trainallmodels():
        decision_tree.train()
        naivebayes.train()
        svm.train()
        knn.train()
        random_forest.train()
        logistic_regression.train()

        if(load_dnn_flag):
            dlseq.check_for_model()
        else:
            print("DNN model not loaded")

        print("Trained all models")
        startscreen.exitstart()
        return
  
class startscreen:
    
    start_window = None

    @staticmethod
    def run():
        print("Start screen")
        import tkinter as tk
        from tkinter import Label

        start_window = tk.Tk()
        start_window.geometry('270x80+670+320')
        start_window.title("Depresso Detecto 3000")
        start_window.overrideredirect(1)
        start_window.after(100, gui.trainallmodels)
        try:
            Label(start_window, text='Loading models\nPlease wait...', font=('CountachWeb', 14, 'normal')).place(x=80, y=15)
        except:
            Label(start_window, text='Loading models\nPlease wait...', font=('arial', 14, 'normal')).place(x=80, y=15)
            print('Countachweb font not found. Using arial.')

        #start_window.protocol("WM_DELETE_WINDOW", sys.exit())
        #tk.Button(start_window, text ="Cancel", command = sys.exit).place(x=150, y=60)
        startscreen.start_window = start_window
        start_window.mainloop()
        return

    @staticmethod
    def exitstart():
        start_window = startscreen.start_window
        start_window.destroy()
        return
      
class homescreen:

    root_window = None
    result = None
    label_text = ""

    @staticmethod
    def get_model_result(str, arglist):
        result = None
        if str == 'Decision tree':
            result  = decision_tree.predict(arglist)
        elif str == 'Naive-Bayes':
            result = naivebayes.predict(arglist)
        elif str == 'SVM':
            result = svm.predict(arglist)
        elif str == 'KNN':
            result = knn.predict(arglist)
        elif str == 'Random forest (%)':
            result = random_forest.predict(arglist)
        elif str == 'Logistic regression':
            result = logistic_regression.predict(arglist)
        elif str == 'Sequential (DNN) (%)':
            result = dlseq.predict(arglist)
        print(result)
        homescreen.result = result
        return

    @staticmethod
    def run():
        import tkinter as tk
        from tkinter import Label, Radiobutton, ttk

        print("Home screen")
        root_window = tk.Tk()
        root_window.geometry('700x450+500+200')
        root_window.title("Depresso Detecto 3000")
        homescreen.root_window = root_window
        try:
            Label(root_window, text='Depresso Detecto 3000', font=('CountachWeb', 18, 'normal')).place(x=240, y=10)
        except:
            Label(root_window, text='Depresso Detecto 3000', font=('arial', 18, 'normal')).place(x=240, y=10)
            print('Countachweb font not found. Using arial.')

        Label(root_window, text='Model:', font=('arial', 10, 'normal')).place(x=30, y=15)

        Label(root_window, text='Environmental satisfaction', font=('arial', 10, 'normal')).place(x=30, y=90)
        Label(root_window, text='Achievement satisfaction', font=('arial', 10, 'normal')).place(x=30, y=130)
        Label(root_window, text='Financial stress', font=('arial', 10, 'normal')).place(x=30, y=170)
        Label(root_window, text='Insomnia', font=('arial', 10, 'normal')).place(x=30, y=210)
        Label(root_window, text='Anxiety', font=('arial', 10, 'normal')).place(x=30, y=250)
        Label(root_window, text='Deprived', font=('arial', 10, 'normal')).place(x=30, y=290)
        Label(root_window, text='Abused', font=('arial', 10, 'normal')).place(x=30, y=330)
        Label(root_window, text='Cheated', font=('arial', 10, 'normal')).place(x=370, y=90)
        Label(root_window, text='Threatened', font=('arial', 10, 'normal')).place(x=370, y=130)
        Label(root_window, text='Suicidal', font=('arial', 10, 'normal')).place(x=370, y=170)
        Label(root_window, text='Inferiority complex', font=('arial', 10, 'normal')).place(x=370, y=210)
        Label(root_window, text='Recent conflict', font=('arial', 10, 'normal')).place(x=370, y=250)
        Label(root_window, text='Recent loss', font=('arial', 10, 'normal')).place(x=370, y=290)

        #this is the declaration of the variable associated with the radio button group
        
        rbEnvsat = tk.IntVar()
        rbPossat = tk.IntVar()
        rbFinstr = tk.IntVar()
        rbInsom = tk.IntVar()
        rbAnx = tk.IntVar()
        rbDepri = tk.IntVar()
        rbAbuse = tk.IntVar()
        rbCheat = tk.IntVar()
        rbThreat = tk.IntVar()
        rbSuic = tk.IntVar()
        rbInfer = tk.IntVar()
        rbConf = tk.IntVar()
        rbLoss = tk.IntVar()
        
        rbvalues = [('Yes', 1, 200), ('No', 0, 260)]
        for text, mode, xpos in rbvalues:
            Radiobutton(root_window, text=text, variable=rbEnvsat, value=mode, font=('arial', 10, 'normal')).place(x=xpos, y=90)
            Radiobutton(root_window, text=text, variable=rbPossat, value=mode, font=('arial', 10, 'normal')).place(x=xpos, y=130)
            Radiobutton(root_window, text=text, variable=rbFinstr, value=mode, font=('arial', 10, 'normal')).place(x=xpos, y=170)
            Radiobutton(root_window, text=text, variable=rbInsom, value=mode, font=('arial', 10, 'normal')).place(x=xpos, y=210)
            Radiobutton(root_window, text=text, variable=rbAnx, value=mode, font=('arial', 10, 'normal')).place(x=xpos, y=250)
            Radiobutton(root_window, text=text, variable=rbDepri, value=mode, font=('arial', 10, 'normal')).place(x=xpos, y=290)
            Radiobutton(root_window, text=text, variable=rbAbuse, value=mode, font=('arial', 10, 'normal')).place(x=xpos, y=330)

        rbvalues = [('Yes', 1, 540), ('No', 0, 620)]
        for text, mode, xpos in rbvalues:
            Radiobutton(root_window, text=text, variable=rbCheat, value=mode, font=('arial', 10, 'normal')).place(x=xpos, y=90)
            Radiobutton(root_window, text=text, variable=rbThreat, value=mode, font=('arial', 10, 'normal')).place(x=xpos, y=130)
            Radiobutton(root_window, text=text, variable=rbSuic, value=mode, font=('arial', 10, 'normal')).place(x=xpos, y=170)
            Radiobutton(root_window, text=text, variable=rbInfer, value=mode, font=('arial', 10, 'normal')).place(x=xpos, y=210)
            Radiobutton(root_window, text=text, variable=rbConf, value=mode, font=('arial', 10, 'normal')).place(x=xpos, y=250)
            Radiobutton(root_window, text=text, variable=rbLoss, value=mode, font=('arial', 10, 'normal')).place(x=xpos, y=290)

        cb_option = tk.StringVar() #combobox
        cb = ttk.Combobox(root_window, textvariable = cb_option, state = 'readonly')
        cb.place(x=30, y=40)

        if(load_dnn_flag):
            cb['values'] = ('Decision tree','Naive-Bayes','SVM','KNN','Random forest (%)','Logistic regression','Sequential (DNN) (%)')
        else:
            cb['values'] = ('Decision tree','Naive-Bayes','SVM','KNN','Random forest (%)','Logistic regression')
        cb.current(0)

        '''menu = Menu(root_window) #menu button to retrain sequential model
        menu.add_command(label='Retrain DNN model', command=dlseq.train)
        root_window.config(menu=menu)'''

        homescreen.root_window = root_window
        
        def getRbValue():

            envsat = rbEnvsat.get()
            possat = rbPossat.get()
            finstr = rbFinstr.get()
            insom = rbInsom.get()
            anx = rbAnx.get()
            depri = rbDepri.get()
            abuse = rbAbuse.get()
            cheat = rbCheat.get()
            threat = rbThreat.get()
            suic = rbSuic.get()
            infer = rbInfer.get()
            conf = rbConf.get()
            loss = rbLoss.get()

            gui.algo_option = cb.get()
            print(gui.algo_option)
            arglist = envsat, possat, finstr, insom, anx, depri, abuse, cheat, threat, suic, infer, conf, loss
            arglist = np.array(arglist).reshape(1,-1)
            print(arglist)
            gui.arglist = arglist
            homescreen.get_model_result(gui.algo_option, gui.arglist)
            result_label = Label(root_window, font=('arial', 10, 'normal'))
            label_text = ''
            label_color = '#000000'
            if homescreen.result[0]==None:
                label_text = "Error!"
                label_color = '#800000'
            elif homescreen.result[0] == 1:
                label_text = "High probability of depression"
                if homescreen.result[1] != -1:
                    label_text = label_text + ": " + str(int(homescreen.result[1]*100)) + "%"
                label_color = '#800000'
            elif homescreen.result[0] == 0:
                label_text = "Low probability of depression"
                if homescreen.result[1] != -1:
                    label_text = label_text + ": " + str(int(homescreen.result[1]*100)) + "%"
                label_color = '#138000'
            result_label = Label(text=label_text, fg = label_color)
            result_label.place(x=370, y=330)
            result_label.after(4000, result_label.destroy)
            return #0,1,1,1,1,1,1,1,1,1,1,0,0

        tk.Button(root_window, text = "Submit", command = getRbValue).place(x=310, y=400) #submit button
        root_window.mainloop()       
        return

gui.main()