package ml.classifiers;

import java.util.ArrayList;

import ml.data.CrossValidationSet;
import ml.data.DataSet;
import ml.data.DataSetSplit;
import ml.data.Example;

public class Experiments {
    public static void main(String[] args){

        DataSet d = new DataSet("data/titanic-train.csv", 0);
        double[] etas = new double[3];
        int[] its = new int[3];
        int[] hid = new int[3];

        etas[0] = 0.1;
        etas[1] = 0.3;
        etas[2] = 0.5;

        its[0] = 100;
        its[1] = 150;
        its[2] = 200;

        hid[0] = 7;
        hid[1] = 8;
        hid[2] = 9;

        for(double eta : etas){
            for(int it : its){
                for(int hidden: hid){

                    System.out.println("Eta: " + eta + ", Iterations: " + it + ", Hidden Num: " + hidden);
                    TwoLayerNN neural = new TwoLayerNN(hidden);
                    neural.setEta(eta);
                    neural.setIterations(it);
    
                    DataSetSplit split = d.split(0.9);

                    DataSet train = split.getTrain();
                    DataSet test = split.getTest();

                    neural.setIterations(200);

                    double trainAccuracy = 0;
                    double testAccuracy = 0;

                    neural.train(train);
                    neural.train(test);

                    ArrayList<Example> testSet = test.getData();
                    ArrayList<Example> trainSet = train.getData();

                    double testCount = 0;
                    double trainCount = 0;

                    for (int h = 0; h < testSet.size(); h++){

                        Double c = neural.classify(testSet.get(h));

                        if (c == testSet.get(h).getLabel()){

                            testCount ++;

                        }

                    }

                    for (int h = 0; h < trainSet.size(); h++){

                        Double c = neural.classify(trainSet.get(h));

                        if (c == trainSet.get(h).getLabel()){

                            trainCount ++;

                        }

                    }


                    testAccuracy = testCount / testSet.size();
                    trainAccuracy = trainCount / trainSet.size();

                    System.out.println("Training accuracy: " + trainAccuracy);
                    System.out.println("Testing accuracy: " + testAccuracy);
                
                
                        }
                    }
        }

        

        //20 50 100
        // 2 6 10

       







        // //1


        // // neural.train(d);

        // neural.initializeNetwork();
        // Example e = new Example();

        // e.addFeature(0, 0.5);
        // e.addFeature(1, 0.2);
        // e.addFeature(2, 1.0);
        // e.setLabel(-1);
        
        // neural.getOutput(e);






        //3

    //     CrossValidationSet crossSet = new CrossValidationSet(d, 10);

    //     for(int j = 0; j < crossSet.getNumSplits(); j++){
    //         DataSetSplit split = crossSet.getValidationSet(j);

    //         DataSet train = split.getTrain();
    //         DataSet test = split.getTest();


    //         TwoLayerNN neural = new TwoLayerNN(j+1);
    //         neural.setEta(0.1);
    //         neural.setIterations(200);



            
            
    //         double trainAccuracy = 0;
    //         double testAccuracy = 0;

    //         neural.train(train);
    //         neural.train(test);

    //         ArrayList<Example> testSet = test.getData();
    //         ArrayList<Example> trainSet = train.getData();

    //         double testCount = 0;
    //         double trainCount = 0;

    //         for (int h = 0; h < testSet.size(); h++){

    //             Double c = neural.classify(testSet.get(h));

    //             if (c == testSet.get(h).getLabel()){

    //                 testCount ++;

    //             }

    //         }

    //         for (int h = 0; h < trainSet.size(); h++){

    //             Double c = neural.classify(trainSet.get(h));

    //             if (c == trainSet.get(h).getLabel()){

    //                 trainCount ++;

    //             }

    //         }


    //         testAccuracy = testCount / testSet.size();
    //         trainAccuracy = trainCount / trainSet.size();

    //         System.out.println("Training accuracy: " + trainAccuracy);
    //         System.out.println("Testing accuracy: " + testAccuracy);
                
    
    
            
    //     }
        

        
       


    // }
}
}
