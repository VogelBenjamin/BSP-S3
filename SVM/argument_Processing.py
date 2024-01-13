from model_Managing import Model_Managing_Unit
import getopt
import sys

HELP = '''RELIABLE SVM
--------------------------
rsvm.py <arguments> <dataset>
-t = train SVM models
-a = check accuracy
-p = make predictions
-o = name of the svm model'''

class Argument_Processing_Unit:

    def __init__(self):
        self.train = False
        self.predict = False
        self.acc = False
        self.model_reference = ""        

    def verify_inputs(self, argv)-> (str,str):
        model_ref, data_ref = "", ""
        opts, args = getopt.getopt(argv, "htpo:a")
        for opt, arg in opts:
            if opt == '-h':
                print(HELP)
                sys.exit()
                return ("", "")
            elif opt == "-t":
                self.train = True
            elif opt == "-a":
                self.acc = True
            elif opt == "-p":
                self.predict = True
            elif opt == "-o":
                self.model_reference = arg
        if not self.model_reference:
            print('No model_reference defined \nmodel_reference set to \'rsvm_model\'')
        if len(args) == 1:
            data_ref = args[0]
        return model_ref, data_ref

    def delegate_task(self, model_manager: Model_Managing_Unit, data_ref : str) -> None:
        X,y = self.process_data(data_ref)
        model_manager.provide_data(X,y)
        model_manager.initialize_environment()
        try: 
            if self.train:
                self.initiate_training(model_manager)
            if self.acc:
                self.initiate_acc_check(model_manager)
            if self.predict:
                self.initiate_prediction(model_manager)
        except ValueError:
            raise ValueError("Dataset is of unexpected shape")

    def process_data(self, data: str) -> (list, list):
        X,y = [],[]
        label = "label" #input("Please provide me with the name of the target feature: ") 
        with open(data, "r") as file:
            line = file.readline()
            labels = line[:-1].split(",")
            label_idx = labels.index(label) if self.train or self.acc or self.predict else -1
            while True:
                line = file.readline()
                if not line:
                    break
                data = line[:-1].split(",")
                lst = []
                for i in range(len(data)):
                    if i != label_idx:
                        lst.append(float(data[i]))
                    else:
                        y.append(data[i])
                X.append(lst)
            file.close()
        return X,y
                
    def store_result(self, result):
        with open(f"{self.model_reference}_result.csv", "w") as file:
            for value in result.values():
                file.write(f"{value}\n")
            file.close()
            
    def initiate_training(self, m_m: Model_Managing_Unit) -> None:
        m_m.train_models()

    def initiate_prediction(self, m_m: Model_Managing_Unit) -> None:
        result = m_m.make_predictions()
        self.store_result(result)
    
    def initiate_acc_check(self, m_m: Model_Managing_Unit) -> None:
        av = m_m.check_accuracy()
        print(av)