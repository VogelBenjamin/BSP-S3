from SVM_Model import SVM
import numpy as np

class Model_Managing_Unit:

    def __init__(self, filename: str):
        self.svm_collection: dict = {}
        self.X: np.ndarray = None
        self.y: np.ndarray = None
        self.label: np.ndarray = None
        self.file_refference: str = filename

    def provide_data(self, X: list, y: list = None) -> None:
        val_to_str = lambda label : str(label)
        self.X = np.array(X)
        self.y = np.array([val_to_str(i) for i in y])
        self.label = np.unique(self.y)

    def initialize_environment(self) -> None:
        for label in self.label:
            self.svm_collection[label] = SVM()
        
    def train_models(self) -> None:
        weight_dict = {}
        for label in self.label:
            curr_SVM: SVM = self.svm_collection[label]

            # initialize the labels to {+1,-1} values
            svm_label = self.__process_labels(self.y, label)
            curr_SVM.set_data(self.X, svm_label)

            # train SVM
            curr_SVM.train()

            # store weigths in dictionary, to be store in file
            weight_dict[label] = curr_SVM.get_weights()
        self.__store_SVM_data(weight_dict)


    def make_predictions(self) -> dict:
        try:
            self.__load__SVM_data()
        except KeyError as err:
            raise err
        results = {}
        outPut = {}
        for datapoint in self.X:
            for label in self.label:
                outPut[label] = self.svm_collection[label].predict(datapoint)
            results[str(list(datapoint))] = max(outPut, key=outPut.get) 
            outPut = {}
        return results


    def check_accuracy(self) -> float:
        r = self.make_predictions()
        c = 0
        correct = 0
        for k, v in r.items():
            if v == self.y[c] : correct += 1
            c += 1
        return correct/self.X.shape[0]

    def __process_labels(self, y: np.ndarray, positive_label) -> np.ndarray:
        mapping_func = lambda label : 1 if label == positive_label else -1
        vectorized_func = np.vectorize(mapping_func)
        processed_y: np.ndarray = vectorized_func(np.copy(y))
        return processed_y

    def __store_SVM_data(self, weigth_info: dict) -> None:
        file = open(f"{self.file_refference}", "w")

        file.write("--Base Features--\n")
        for i in range(self.X.shape[0]):
            file.write(f"{str(list(self.X[i])).replace(',','')},{self.y[i]}\n")
        
        file.write("w and b:\n")
        for k,v in weigth_info.items():
            file.write(f"{k}\n{str(list(v[0])).replace(',','')}\n{v[1]}\n{v[2]}\n")
        file.close()

    def __load__SVM_data(self) -> None:
        file = open(f"{self.file_refference}", "r")

        X, Y = [], []
        w = []
        b = None
        mult = []

        line = file.readline()
        line = file.readline()
        while(line != "w and b:\n"):
            data = line.split(",")
            x,y = [float(x) for x in data[0][1:-1].split(" ") if x != ""], data[1][:-1]
            X.append(x)
            Y.append(y)
            line = file.readline()
        self.label = np.unique(Y)
        self.initialize_environment()
        line = file.readline()
        while(line):
            label = line[:-1]
            line = file.readline()
            w = [float(x) for x in line[1:-2].split(" ") if x != ""]
            
            line = file.readline()
            b = float(line)
            line = file.readline().replace("[","")
            while True:
                valArr = [float(x) for x in line.replace("\n","").replace("]","").split(" ") if x != ""]
                for item in valArr:
                    mult.append(item)
                if "]" in line:
                    break
                line = file.readline()
            try:
                n_Y = Y.copy()
                
                for i in range(len(n_Y)):
                    if n_Y[i] == label: n_Y[i] = 1
                    else: n_Y[i] = -1
                self.svm_collection[label].set_data(np.array(X),np.array(n_Y))
                self.svm_collection[label].set_weights(np.array(w),b, np.array(mult))
            except KeyError:
                raise KeyError("It seems that the provided datas features do not correlate with the models features")
            mult = []
            line = file.readline()

