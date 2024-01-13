from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
import time
X,y = [],[]

with open("skData.csv","w") as new_file:
    new_file.write("dataSize,time\n")
    for num in range(100,2001,100):
        start = time.time()
        with open(f"datasets/circles_{num}_0.09_0.5.csv") as file:
            line = file.readline()
            line = file.readline()
            while line:
                dt = line.split(",")
                X.append([float(dt[0]),float(dt[1])])
                y.append(int(dt[2]))
                line = file.readline()
            file.close()
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)
        svc = SVC()
        svc.fit(X_train,y_train)
        end = time.time()
        new_file.write(f"{num},{end-start}\n")
    
    
    new_file.close()
    
