import numpy
import scipy.special #for sigmoid func expit()
import matplotlib.pyplot
%matplotlib inline   #for jupyter notebook / colab

class neuralNetwork:

  #initialise the neural network
  def __inti__(self,inputnodes,hiddennodes,outputnodes,learningrate):
    #set number of nodes in each input,hidden,output layer

    self.inodes = inputnodes
    self.hnodes = hiddennodes
    self.onodes = outputnodes

    #link weight matrices , wih and who
    #weights inside the arrays are w_i_j, where link is from node i to node j in the next layer , w11 , w21 etc
    self.win = numpy.random.normal(0.0,pow(self.hnodes,-0.5),(self.hnodes,self.inodes))
    self.who = numpy.random.normal(0.0, pow(self.onodes, -0.5),(self.onodes, self.hnodes))
    
    #learning rate
    self.lr = learningrate

    #activation function sigmoid using lamda x
    self.activation_function = lambda x : scipy.special.expit(x)

    pass

    def query(self,inputs_list):

        #convert input list into 2d array
        inputs = numpy.array(inputs_list ,ndmin=2).T

        #calculate signals into hidden layer
        hidden_input = numpy.dot(self.wih,inputs)
        #calculate the signals from hidden layer
        hidden_output = self.activation_function(hidden_input)

        #calculate signals into final output layer
        final_inputs = numpy.dot(self.who,hidden_output)
        #calculate signals from final output layer
        final_output = self.activation_function(final_inputs)

        return final_output

    def train():
        pass
        #to be added
    
