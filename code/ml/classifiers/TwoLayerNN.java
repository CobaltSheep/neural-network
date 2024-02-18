package ml.classifiers;

import java.util.ArrayList;
import java.util.Random;
import java.util.Set;

import ml.data.DataSet;
import ml.data.Example;


/**
 * @author Pete Boyle and Luca Snoey
 * Assignment 8
 */
public class TwoLayerNN implements Classifier{

    private int hiddenLayers;
    
    private double eta;

    private int iterations;

    private int numFeatures;

    private DataSet data;

    private double[][] weightsMatrix;

    private double[] outputWeights;

    private double[] inputMatrix;

    private double[] hiddenList;
    

    /**
     * Contructor that intitializes our hyperparameters. Eta is set at 0.1
     * and iterations at 200
     * 
     * @param hl how many hidden layers we will build the network with.
     */
    public TwoLayerNN(int hl){
        this.hiddenLayers = hl;
        this.eta = 0.1;
        this.iterations = 200;
    }

    /**
     * Sets eta 
     * @param e eta number
     */
    public void setEta(double e){
        this.eta = e;
    }

    /**
     * Sets iterations
     * @param i iterations number 
     */
    public void setIterations(int i){
        this.iterations = i;
    }

    /**
     * Makes the weight matrices with random values between -0.1 and 0.1
     * Does this for output weights as well.  
     */
    private void initializeNetwork(){

        // make the weights matrix
        this.weightsMatrix = new double[hiddenLayers][numFeatures];

        // populate the matrix with random vals between -0.1 and 0.1
        for(int i = 0; i<weightsMatrix.length; i++){
            for(int k = 0; k<weightsMatrix[0].length; k++){
                Random a = new Random();
                double randValue = -0.1 * (0.2 * a.nextDouble());
                weightsMatrix[i][k] = randValue;
            }
        }

        // create output weights array and account for added bias feature
        this.outputWeights = new double[hiddenLayers + 1];

        // populate the output weights
        for(int i = 0; i<outputWeights.length; i++){
            Random a = new Random();
            double randValue = -0.1 * (0.2 * a.nextDouble());
            outputWeights[i] = randValue;
        }
        
    }

    /**
     * Multiplication of two matrices (dot product)
     * @param list1 list of doubles
     * @param list2 list of doubles
     * @return
     */
    private double dotProduct(double[] list1, double[] list2){
        
        double sum = 0;
        for(int i = 0; i<list1.length; i++){
            sum+= list1[i] * list2[i];
        }
        return sum;
    }


    /**
     * Trains the neural network given a dataset that gets passed in. 
     * Does this by running through examples and updating weights and 
     * hidden layer and output values. It does that process for a set
     * amount of iterations.
     * @param data Dataset to train 
     */
    @Override
    public void train(DataSet data) {
        //copy the data with bias
        this.data = data.getCopyWithBias();


        initializeNetwork();
        

        ArrayList<Example> biasedExamples = data.getData();
        
        // loop through number of iterations
        for(int k=0; k<iterations; k++){

            // go through each example
            for(Example e : biasedExamples){

                // use helper method to generate output for the example
                double output = getOutput(e);

                // Calculate the error for the output layer
                double error = (e.getLabel() - output) * (1 - output * output);
                

                // Backpropagation: update output layer weights
                for (int i = 0; i < outputWeights.length; i++) {
                    outputWeights[i] += eta * error * hiddenList[i];
                }


                // Backpropagation: update hidden layer weights
                for (int h = 0; h < hiddenLayers; h++) {
                    for (int j = 0; j < numFeatures; j++) {
                        double loss = (1 - hiddenList[h] * hiddenList[h]) * outputWeights[h] * error;
                        weightsMatrix[h][j] += eta * loss * inputMatrix[j];
                    }
                }
            }
        }
        
    }
    
    /**
     * Gets the output value for a given example. Used as a helper method 
     * in train(). Takes in an example and uses the tanh function and 
     * hidden node values and output weights to produce an output. 
     * @param e example from a dataset 
     * @return Double output 
     */
    private double getOutput(Example e){

        // populate input matrix with features
        for(int i = 0; i<numFeatures; i++){
                
            inputMatrix[i] = e.getFeature(i); 
                    
        }
               
        // create hidden list accounting for added bias feature
        hiddenList = new double[hiddenLayers + 1];

        // populate hidden list with the dot products, unless it is last feature, in that case it is just bias of 1
        for(int i = 0; i<hiddenList.length; i++){
            if(i == hiddenList.length - 1){
                hiddenList[i] = 1;
            } else{
                hiddenList[i] = Math.tanh(dotProduct(weightsMatrix[i], inputMatrix));
            }
            
        }

        double output = Math.tanh(dotProduct(hiddenList, outputWeights));

        return output;

    }

    /**
     * Classifies a given example from a dataset. Returns 1.0 if the output
     * is positive, and -1.0 if the output is negative. 
     * @param example an example consisting of features and a label
     */
    @Override
    public double classify(Example example) {

        // add the bias feature each example should have in training 
        example = data.addBiasFeature(example);

        if(getOutput(example) > 0){
            return 1.0;
        }
        else{
            return -1.0;
        }
    }

    /**
     * Gives a confidence value given an example from a dataset.
     * In this instance is is the absolute value of the output of the 
     * example (its distance from 0).
     * @param example an example consisting of features and a label 
     * 
     */
    @Override
    public double confidence(Example example) {
        return Math.abs(getOutput(example));
    }
    
}