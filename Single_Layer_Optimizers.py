import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score

class Single_Layer_Neuron:
    
    def __init__(self, data):
        self.train_data = data
    
    # Split train_data, train_target, test_data, test_target
    def create_val_set(self, test_frac = 0.2):
        np.random.shuffle(self.train_data)
        split = int(len(self.train_data)*(test_frac))
        self.val_data = self.train_data[0:split, :]
        self.train_data = self.train_data[split:, :]
        
    # Calculate the standardization metrics from the training data
    def get_standardization_metrics(self):
        self.data_means = np.mean(self.train_data, axis=0)
        self.data_scale = np.std(self.train_data, axis=0)
        
    def standardize(self, data):
        return (data - self.data_means) / self.data_scale
    
    # Add ones column on the left hand side for the bias term
    def add_ones_column(self, data):
        return np.column_stack((np.ones([len(data)]), data))

    def hypothesis_function(self, data):
        raise NameError('hypothesis_function is not defined for this class!')

    def PreprocessData(self):
    	# Preprocess the data: Create validation set, Split off targets, Standardize, Add ones column
        if self.val_set_fraction:
            self.create_val_set(self.val_set_fraction)
        self.train_target = self.train_data[:, -1]
        self.train_data = self.train_data[:, 0:-1]
        self.get_standardization_metrics()
        self.train_data = self.standardize(self.train_data)
        self.train_data = self.add_ones_column(self.train_data)
        if self.val_set_fraction:
            self.val_target = self.val_data[:, -1]
            self.val_data = self.val_data[:, 0:-1]
            self.val_data = self.standardize(self.val_data)
            self.val_data = self.add_ones_column(self.val_data)
    
    def Calc_Cost(self, data, target):
        raise NameError('Calc_Cost is not defined for this class!')
        
    def Gradient_Descent(self, data, target, alpha):
        raise NameError('Gradient_Descent is not defined for this class!')
    
    # Fit the weights to the data using gradient descent
    def fit(self, alpha = 0.01, iterations = 100, init = 'rand', val_set_fraction = 0, reg_coef = 0):
        self.alpha = alpha
        self.iterations = iterations
        self.val_set_fraction = val_set_fraction
        self.reg_coef = reg_coef

        # Preprocess the data: Create validation set, Split off targets, Standardize, Add ones column
        self.PreprocessData()
        
        # Initialize the weight parameters using the chosen scheme
        if init == 'rand':
            self.weights = np.random.randn(self.train_data.shape[1])
        elif init == 'zeros':
            self.weights = np.zeros(self.train_data.shape[1])
        else:
            print('Invalid or missing init scheme parameter')
            return
        
        # Create vectors to store cost values during training
        self.cost_train = np.zeros(self.iterations)
        if self.val_set_fraction:
            self.cost_val = np.zeros(self.iterations)
        
        # Gradient descent through data using full batches
        for i in range(iterations):
            # First record the cost at each iteration
            self.cost_train[i] = self.Calc_Cost(self.train_data, self.train_target)
            if self.val_set_fraction:
                self.cost_val[i] = self.Calc_Cost(self.val_data, self.val_target)
            # Update weights
            self.Gradient_Descent(self.train_data, self.train_target, self.alpha)
        
    # Show how cost changed during training
    def plot_cost(self):
        plt.plot(range(self.iterations), self.cost_train)
        if self.val_set_fraction:
            plt.plot(range(self.iterations), self.cost_val)
            plt.legend(('Training Data Cost', 'Validation Data Cost'), loc = 'best')
        plt.xlabel('Batch Iteration')
        plt.ylabel('Cost')
        plt.title('Cost During Training')
        plt.show()
        
    # Use current weights to predict either the training, validation, or a newly provided test dataset
    def predict(self, dataset = 'train', test_data = None):
        
        # Setup data according to dataset type
        if dataset == 'train':
            self.prediction_dataset = self.train_data
            self.prediction_target = self.train_target
        elif dataset == 'val':
            if self.val_set_fraction:
                self.prediction_dataset = self.val_data
                self.prediction_target = self.val_target
            else: print('Error: No validation set')
        elif dataset == 'test':
            # Confirm that valid test data was provided
            if type(test_data) != np.ndarray:
                print("Error: Must supply test dataset of type ndarray as second argument.")
                return
            else:
                self.test_data = test_data
                # If test targets were provided, split them off
                if self.test_data.shape[1] == self.train_data.shape[1]+1:
                    self.test_target = self.test_data[:,-1]
                    self.prediction_target = self.test_target
                    self.test_data = self.test_data[:,:-1]
                # Preprocess test data
                self.test_data = self.standardize(self.test_data)
                self.test_data = self.add_ones_column(self.test_data)
                self.prediction_dataset = self.test_data
        else:
            print("First argument must be a valid dataset type: 'train', 'val', or 'test'")
            return
        
        # Predict Data
        self.predictions = self.hypothesis_function(self.prediction_dataset)
        
    # Plot target and predictions vs feature
    def single_feature_plot(self, feature = 1):
        plt.plot(self.prediction_dataset[:,feature], self.prediction_target, 'bx', \
                 self.prediction_dataset[:,feature], self.predictions, 'rx')
        plt.xlabel('Feature {}'.format(feature))
        plt.ylabel('Y Value')
        plt.legend(('Input Data', 'Predictions'), loc = 'best')
        plt.title('Output vs. Individual Feature')
        plt.show()
        
    # Calculate the fraction of variance in targets that is explained by the predictions
    def Variance_Explained(self):
        print("Fraction of variance explained is ", r2_score(self.prediction_target, self.predictions))

class Linear_Regression(Single_Layer_Neuron):
    
    def hypothesis_function(self, data):
        return np.dot(data, self.weights)
    
    def Calc_Cost(self, data, target):
        return np.square((np.dot(data, self.weights) - target)).sum()/2/data.shape[0] + \
               self.reg_coef/2/data.shape[0] * np.square(self.weights[1:]).sum()
    
    def Gradient_Descent(self, data, target, alpha):
        self.weights = self.weights-alpha/data.shape[0] * \
                       (np.dot(data.transpose(),self.hypothesis_function(data)-target) + \
                        np.concatenate(([0], self.reg_coef*self.weights[1:])))

    # Plot predictions against targets
    def target_prediction_plot(self):
        plt.plot(self.prediction_target, self.predictions, 'o')
        plt.xlabel('Targets')
        plt.ylabel('Predictions')
        plt.title('Predictions vs. Targets')
        plt.show()
    
    # Calculate the weights using the Normal Equation
    def NormalEquation(self):
        self.NE_weights = np.matmul(np.linalg.inv(np.matmul(self.train_data.transpose(), self.train_data)), \
                                    np.matmul(self.train_data.transpose(), self.train_target))
        print("Normal Equation results in weights of ", self.NE_weights)

class Logistic_Regression(Single_Layer_Neuron):

    def hypothesis_function(self, data):
        return 1 / (1 + np.exp(-np.dot(data, self.weights)))
    
    def Calc_Cost(self, data, target):
        return -1/data.shape[0] * np.sum( target*np.log(self.hypothesis_function(data)) + \
                                         (1-target)*np.log(1 - self.hypothesis_function(data)) ) + \
               self.reg_coef/2/data.shape[0] * np.square(self.weights[1:]).sum()
        
    def Gradient_Descent(self, data, target, alpha):
    	self.weights = self.weights-alpha/data.shape[0] * \
                       (np.dot(data.transpose(),self.hypothesis_function(data)-target) + \
                        np.concatenate(([0], self.reg_coef*self.weights[1:])))

    def Classification_Accuracy(self):
    	self.classification_accuracy = np.average(np.rint(self.predictions) == self.prediction_target.astype(int))
    	print('My classification accuracy is ', self.classification_accuracy)