import numpy as np
from random import randrange,choice
class SVM:
    '''
    SMO algorithm produced using the Pseudo Code provided by:
    https://www.microsoft.com/en-us/research/wp-content/uploads/2016/02/tr-98-14.pdf
    '''

    def __init__(self, c = 1, kernel = "gaussian", tol = 0.001):
        
        kernel_dict = {
            "linear" : self.linear_kernel, 
            "gaussian" : self.gaussian_kernel
        }

        self.X = None
        self.y = None
        self.m, self.n = None, None
        self.multipliers = None
        self.w = None
        self.b = 0
        self.C = c
        self.tol = tol
        self.linear = True if kernel == "linear" else False
        self.kernel = kernel_dict[kernel]
        self.epsilon = 0.001
        self.error_cache = None
        self.ready = False

    def set_data(self, X, y):
        self.X = X
        self.y = y
        self.m, self.n = X.shape
        self.multipliers = np.zeros(self.m)
        self.w = np.zeros(self.n)
        self.error_cache = np.zeros(self.m)
        self.ready = True

    def set_hyper_param(self, c, linear, kernel, tol):
        self.C = c
        self.tol = tol
        self.linear = True if kernel == "linear" else False
        self.kernel = self.kernel_dict[kernel]
        self.epsilon = 0.001

    def check_data(self):
        if not self.ready:
            raise Exception("Missing Data Input")

    def train(self):
        '''
        Trains the SVM
        :Pre-Condition: data and hyperparameters have to be already set
        :return: None
        '''
        self.check_data()

        numChanged = 0 
        examineAll = True

        # modify multiplier pairs until the entire set  
        # of multipliers satisfy the KKT conditions
        while numChanged > 0 or examineAll:
            numChanged = 0
            
            # determine whether to choose the first multiplier from
            # the entire set or only from non-bounded mutliplier set
            if examineAll:
                for i in range(self.m):
                    numChanged += self.examineExample(i)
            else:
                non_bound_lst = self.non_bound_index()
                for idx in non_bound_lst:
                    numChanged += self.examineExample(idx)

            # case 1: if the entire set of multipliers has been examined
            # the next iterations only examine the non-bound multipliers
            # case 2: if only the non-bound multipliers have been examined
            # and none have been changed, examine all multipliers again
            if examineAll:
                examineAll = False
            elif numChanged == 0:
                examineAll = True
            
            # while loop is exited when the entire set of multipliers has 
            # been examined and no multipliers have been modified -> KKT satisfied
    
    def predict(self, feature):
        self.check_data
        if self.linear:
            return float(np.dot(self.w,feature)) - self.b
        else: 
            return np.sum([self.multipliers[i]*self.y[i]*self.kernel(self.X[i],feature) for i in range(self.m)]) - self.b
        
    def accuracy(self, test_X, test_y):
        count = 0
        for i in range(test_X.shape[0]):
            if np.sign(self.predict(test_X[i])) == test_y[i]:
                count += 1
        return count/test_X.shape[0]
    
    def get_weights(self):
        return self.w, self.b, self.multipliers
    
    def set_weights(self, new_w, new_b, new_multipliers):
        self.w, self.b, self.multipliers = new_w, new_b, new_multipliers
        
    def non_bound_index(self):
        lst = []
        for i in range(self.m):
            if 0 < self.multipliers[i] and self.multipliers[i] < self.C: 
                lst.append(i)
        return lst
    
    def examineExample(self, i2):
        '''
        Takes a multiplier, searches for an optimal second multiplier to then
        perform coordinate descent and informs the calling function whether 
        the multipliers have been improved or not

        :param i2: index of the first chosen multiplier
        :return: 1 if multipliers have been optimized, 0 otherwise
        '''
        # get/set information of the data point related to lambda_2
        self.y2 = self.y[i2]
        self.l2 = self.multipliers[i2]
        self.X2 = self.X[i2]
        self.E2 = self.get_error(i2)
        self.r2 = self.E2*self.y2

        # if the error out of tolerance bounds and the mutliplier is within bounds
        # continue with the function and find lambda_1 for coordinate descent
        if not ((self.r2 < -self.tol and self.l2 < self.C) or (self.r2 > self.tol and self.l2 > 0)):
            return 0
        
        # search for multiplier from non-bounded set of multipliers
        non_bound_idx = self.non_bound_index()
        if len(non_bound_idx):

            # search for optimal second multiplier by looking for multipliers that
            # maximize the value of |E_1 - E_2| 
            i1 = self.chose_second_mult(non_bound_idx, self.E2)
            if self.coordinateDescent(i1,i2):
                return 1

            # if previous multiplier does not result in change, search for 
            # a new multiplier randomly from the set of non-bound multipliers
            rand_i = randrange(len(non_bound_idx))
            for i1 in non_bound_idx[rand_i:] + non_bound_idx[:rand_i]:
                if self.coordinateDescent(i1,i2):
                    return 1
        
        # if non of the non-bound multipliers result in change, 
        # try out all multipliers, chosing them randomly
        all_indices = list(set(range(self.m)).difference(set(non_bound_idx)))
        rand_i = choice(all_indices)
        for i1 in all_indices[rand_i:0] + all_indices[:rand_i]:
            if self.coordinateDescent(i1,i2):
                return 1
            
        return 0

    def get_error(self,idx):
        if 0 < self.multipliers[idx] < self.C:
            return self.error_cache[idx]
        else:
            return self.predict(self.X[idx]) - self.y[idx]
    
    def chose_second_mult(self, non_bound_indices, E2):
        '''
        Take a set of indices and returns the index of the multiplier
        that maximizes |E_1 - E_2| 

        :param non_bound_indicies: set of all indexes of non-bound multipliers
        :param E2: the error of the first chosen multiplier
        :return: index of the multiplier maxmimizing absolute error.
        '''
        i1 = -1
        maxi = 0

        for idx in non_bound_indices:
            E1 = self.error_cache[idx] - self.y[idx]
            step = abs(E1 - E2)
            if step > maxi:
                maxi = step
                i1 = idx
        return i1
    
    def coordinateDescent(self, i1, i2):
        '''
        Takes the indecies of two multipliers and determines optimal values
        to minimize the objective function of the SVM optimization problem 
        and returns whether the multipliers have been modified

        :param i1: index of first multiplier
        :param i2: index of second multiplier
        :return: Boolean indicating whether the multipliers have been optimized
        '''
        # return if the method received identical multipliers
        if i1 == i2: return False

        # get/set information of the data point related to lambda_1
        l1 = self.multipliers[i1]
        y1 = self.y[i1]
        X1 = self.X[i1]
        E1 = self.get_error(i1)
        s = y1*self.y2

        # determine lower nad upper bound
        if y1 != self.y2:
            lower_bound = max(0,self.l2 - l1)
            upper_bound = min(self.C, self.C + self.l2 - l1)
        else:
            lower_bound = max(0,self.l2 + l1 - self.C)
            upper_bound = min(self.C, self.l2 + l1)
        
        if lower_bound == upper_bound:
            return False
        
        # compute k = sec_deriv
        k11 = self.kernel(X1,X1)
        k12 = self.kernel(X1,self.X[i2])
        k22 = self.kernel(self.X[i2],self.X[i2])
        
        sec_deriv = k11 - 2*k12 + k22

        # compute optimal value for lambda_2 and clip it to bounds
        if sec_deriv > 0:
            optimal_l2 = self.l2 + self.y2*(E1 - self.E2)/sec_deriv
            if optimal_l2 < lower_bound: optimal_l2 = lower_bound
            elif optimal_l2 > upper_bound: optimal_l2 = upper_bound
        # if k is negative use other method to determine optimal lambda_2
        else:
            f1 = y1*(E1+self.b)-l1*k11-s*self.l2*k12
            f2 = self.y2*(self.E2+self.b)-s*l1*k12-self.l2*k22
            L1 = l1 + s*(self.l2-lower_bound)
            H1 = l1 + s*(self.l2-upper_bound)
            wL = L1*f1 + lower_bound*f2 + 0.5*(L1**2)*k11 + 0.5*(lower_bound**2)*k22+s*lower_bound*L1*k12
            wH = H1*f1 + upper_bound*f2 + 0.5*(H1**2)*k11 + 0.5*(upper_bound**2)*k22+s*upper_bound*H1*k12

            if wL < wH - self.epsilon:
                optimal_l2 = lower_bound
            elif wL > wH + self.epsilon:
                optimal_l2 = upper_bound
            else:
                optimal_l2 = self.l2

        # if the change in lambda_2 is negligable, return without modifying 
        # the multipliers
        if abs(optimal_l2-self.l2) < self.epsilon*(optimal_l2+self.l2+self.epsilon):
            return False
        
        # determine optimal lambda_1 using optimal lambda_2
        optimal_l1 = l1 + s*(self.l2-optimal_l2)

        # determine improved b
        optimal_b = self.compute_b(E1, l1, optimal_l1, optimal_l2, k11, k12, k22, y1)
        deltab = optimal_b - self.b
        self.b = optimal_b
        
        # determine improved w if linear
        if self.linear:
            optimal_w = self.compute_w()
            self.w = optimal_w

        # calculate new error values
        new_error_1 = optimal_l1 - l1
        new_error_2 = optimal_l2 - self.l2

        # update the error cache
        for i in range(self.m):
            if 0 < self.multipliers[i] < self.C:
                self.error_cache[i] += y1 * new_error_1 * self.kernel(X1, self.X[i]) + self.y2 * new_error_2 * self.kernel(self.X2, self.X[i]) - deltab

        self.error_cache[i1] = 0
        self.error_cache[i2] = 0

        # update the multipliers
        self.multipliers[i1] = optimal_l1
        self.multipliers[i2] = optimal_l2
        return True

    def compute_w(self):
        result = np.zeros(len(self.X[0]))
        for i in range(len(self.y)):
            result = np.add(result, self.multipliers[i] * self.y[i] * self.X[i])
        return result
    
    def compute_b(self, E1, l1, op_l1, op_l2, k11, k12, k22 ,y1):
        if (0 < op_l1) and (op_l1 < self.C):
            return E1 + y1*(op_l1-l1)*k11 + self.y2*(op_l2-self.l2)*k12 + self.b
        elif (0 < op_l2) and (op_l2 < self.C):
            return self.E2 + y1*(op_l1-l1)*k12 + self.y2*(op_l2 - self.l2)*k22 + self.b
        else:
            return ((E1 + y1*(op_l1-l1)*k11 + self.y2*(op_l2-self.l2)*k12 + self.b) + \
                    (self.E2 + y1*(op_l1-l1)*k12 + self.y2*(op_l2 - self.l2)*k22 + self.b))/2.0
    
    def linear_kernel(self, x1, x2):
        return np.dot(x1,x2)
    
    def gaussian_kernel(self, x1, x2,sigma = 1):
        return np.exp(-sigma*np.linalg.norm(x1-x2)**2)

svm = SVM()

