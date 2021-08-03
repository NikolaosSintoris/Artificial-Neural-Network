package src;

import java.io.BufferedReader;
import java.io.FileReader;
import java.io.IOException;
import java.lang.Math;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.Random;

public class MultiLayerPerceptron 
{
	private static final float MIN_RANDOM_VALUE = -1;
	private static final float MAX_RANDOM_VALUE = 1;
	
	private int inputDimension; // d
	private int numberOfCategories; // K
	private int numberOfFirstHiddenLayerNeurons; // H1	
	private int numberOfSecondHiddenLayerNeurons; // H2
	private int numberOfOutputLayerNeurons;
	private String secondHiddenLayerActivationFunction; // linear or hyperbolic
	
	private float[] inputToFirstHiddenLayerWeightsArray;
	private float[] firstToSecondHiddenLayerWeightsArray;
	private float[] secondToOutputLayerWeightsArray;
	
	private float[] firstHiddenLayerBiasArray;
	private float[] secondHiddenLayerBiasArray;
	private float[] outputLayerBiasArray;
	
	private float[] firstHiddenLayerOutput;
	private float[] secondHiddenLayerOutput;
	private float[] neuralNetworkOutput;
	
	private float totalSquareError;
	private float previousTotalSquareError;
	
	private float[] outputLayerDelta;
	private float[] secondHiddenLayerDelta;
	private float[] firstHiddenLayerDelta;
	
	private ArrayList<Float> derivativeErrorInputToFirst;
	private ArrayList<Float> derivativeErrorFirstToSecond;
	private ArrayList<Float> derivativeErrorSecondToOutput;
	private ArrayList<Float> derivativeErrorFirstLayerBias;
	private ArrayList<Float> derivativeErrorSecondLayerBias;
	private ArrayList<Float> derivativeErrorOutputLayerBias;
	
	//private ArrayList<float[]> trainingArrayList;
	private ArrayList<float[]> testArrayList;
	private HashMap<float[], float[]> trainingSetHashMap;
	private HashMap<float[], float[]> testSetHashMap;
	
	private ArrayList<float[]> bestTestSetArrayList;
	
	public MultiLayerPerceptron(int inputDimension, int numberOfCategories, int numberOfFirstHiddenLayerNeurons, int numberOfSecondHiddenLayerNeurons, String secondHiddenLayerActivationFunction)
	{
		this.inputDimension = inputDimension;
		this.numberOfCategories = numberOfCategories;
		this.numberOfFirstHiddenLayerNeurons = numberOfFirstHiddenLayerNeurons;
		this.numberOfSecondHiddenLayerNeurons = numberOfSecondHiddenLayerNeurons;
		this.numberOfOutputLayerNeurons = numberOfCategories;
		this.secondHiddenLayerActivationFunction = secondHiddenLayerActivationFunction;
		
		this.inputToFirstHiddenLayerWeightsArray = new float[this.inputDimension * this.numberOfFirstHiddenLayerNeurons];
		this.firstToSecondHiddenLayerWeightsArray = new float[this.numberOfFirstHiddenLayerNeurons * this.numberOfSecondHiddenLayerNeurons];
		this.secondToOutputLayerWeightsArray = new float[this.numberOfSecondHiddenLayerNeurons * this.numberOfOutputLayerNeurons];
		
		this.firstHiddenLayerBiasArray = new float[this.numberOfFirstHiddenLayerNeurons];
		this.secondHiddenLayerBiasArray = new float[this.numberOfSecondHiddenLayerNeurons];
		this.outputLayerBiasArray = new float[this.numberOfOutputLayerNeurons];
		
		this.firstHiddenLayerOutput = new float[this.numberOfFirstHiddenLayerNeurons];
		this.secondHiddenLayerOutput = new float[this.numberOfSecondHiddenLayerNeurons];
		this.neuralNetworkOutput = new float[this.numberOfOutputLayerNeurons];
		
		this.totalSquareError = 0f;
		
		this.outputLayerDelta = new float[this.numberOfOutputLayerNeurons];
		this.secondHiddenLayerDelta = new float[this.numberOfSecondHiddenLayerNeurons];
		this.firstHiddenLayerDelta = new float[this.numberOfFirstHiddenLayerNeurons];

		//this.trainingArrayList = new ArrayList<float[]>();
		this.testArrayList = new ArrayList<float[]>();
		this.trainingSetHashMap = new HashMap<float[], float[]>();
		this.testSetHashMap = new HashMap<float[], float[]>();
		this.bestTestSetArrayList = new ArrayList<float[]>();
	}
	
	
	public void loadTrainingSet()
	{
        try 
        {
            FileReader fileReader = new FileReader("TrainingSet.txt");
            BufferedReader bufferedReader = new BufferedReader(fileReader);
            String line;
            while ((line = bufferedReader.readLine()) != null) 
            {
            	String[] tempStringArray= line.split(",");
            	float x1 = Float.parseFloat(tempStringArray[0]);
            	float x2 = Float.parseFloat(tempStringArray[1]);
            	float[] example = {x1, x2};
            	
            	int exampleCategory = Integer.parseInt(tempStringArray[2]);
            	
            	float[] targetOutput = codingCategories(exampleCategory);
            	this.trainingSetHashMap.put(example, targetOutput);
            }
            fileReader.close();
        } 
        catch (IOException e) 
        {
            e.printStackTrace();
        }
	}
	
	public void loadTestSet()
	{
        try 
        {
            FileReader fileReader = new FileReader("TestSet.txt");
            BufferedReader bufferedReader = new BufferedReader(fileReader);
            String line;
            while ((line = bufferedReader.readLine()) != null) {
            	String[] tempStringArray= line.split(",");
            	float x1 = Float.parseFloat(tempStringArray[0]);
            	float x2 = Float.parseFloat(tempStringArray[1]);
            	float[] example = {x1, x2};
            	
            	int exampleCategory = Integer.parseInt(tempStringArray[2]);
            	this.testArrayList.add(example);
            	
            	float[] targetOutput = codingCategories(exampleCategory);
            	this.testSetHashMap.put(example, targetOutput);
            }
            fileReader.close();
        } 
        catch (IOException e) 
        {
            e.printStackTrace();
        }
	}
	
	public float[] codingCategories(int category)
	{
		float[] targetOutput = new float[this.numberOfCategories];
    	if(category == 1)
    	{
    		targetOutput[0] = 1;
    		targetOutput[1] = 0;
    		targetOutput[2] = 0;
    	}
    	else if(category == 2)
    	{
    		targetOutput[0] = 0;
    		targetOutput[1] = 1;
    		targetOutput[2] = 0;
    	}
    	else
    	{
    		targetOutput[0] = 0;
    		targetOutput[1] = 0;
    		targetOutput[2] = 1;
    	}
    	return targetOutput;
	}	

	
	public void gradientDescentTraining(int L, float learningRate)
	{
		initializeWeightsRandomly();

		int epoch = 1;
		boolean flag = true;
		while(flag == true)
		{
			//System.out.println("Epoch: " + epoch);
			this.previousTotalSquareError = this.totalSquareError;
			this.totalSquareError = 0f;

			ArrayList<Float> tempDerivativeErrorInputToFirst = new ArrayList<Float>();
			ArrayList<Float> tempDerivativeErrorFirstToSecond = new ArrayList<Float>();
			ArrayList<Float> tempDerivativeErrorSecondToOutput = new ArrayList<Float>();
			ArrayList<Float> tempDerivativeErrorFirstLayerBias = new ArrayList<Float>();
			ArrayList<Float> tempDerivativeErrorSecondLayerBias = new ArrayList<Float>();
			ArrayList<Float> tempDerivativeErrorOutputLayerBias = new ArrayList<Float>();
			
			initializeTempDerivativeErrors(tempDerivativeErrorInputToFirst, tempDerivativeErrorFirstLayerBias, tempDerivativeErrorFirstToSecond, tempDerivativeErrorSecondLayerBias, tempDerivativeErrorSecondToOutput, tempDerivativeErrorOutputLayerBias);

			initializeDerivativeErrorsWithZeros();
			
			int counter = 1;
			for(float[] example: this.trainingSetHashMap.keySet())
			{
				backPropagation(example, this.trainingSetHashMap.get(example));
				
				updateDerivativeErrors(tempDerivativeErrorInputToFirst, tempDerivativeErrorFirstLayerBias, tempDerivativeErrorFirstToSecond, tempDerivativeErrorSecondLayerBias, tempDerivativeErrorSecondToOutput, tempDerivativeErrorOutputLayerBias);
				
				if(counter == L)
				{
					counter = 1;
					updateWeights(learningRate, tempDerivativeErrorInputToFirst, tempDerivativeErrorFirstLayerBias, tempDerivativeErrorFirstToSecond, tempDerivativeErrorSecondLayerBias, tempDerivativeErrorSecondToOutput, tempDerivativeErrorOutputLayerBias);
				
					setTempDerivativeErrors(tempDerivativeErrorInputToFirst, tempDerivativeErrorFirstLayerBias, tempDerivativeErrorFirstToSecond, tempDerivativeErrorSecondLayerBias, tempDerivativeErrorSecondToOutput, tempDerivativeErrorOutputLayerBias);

					initializeDerivativeErrorsWithZeros();
				}
				else
				{
					counter++;
				}
			}
			
			if(epoch > 600)
			{
				if(Math.abs(this.totalSquareError - this.previousTotalSquareError) <= 0.1)
				{
					flag = false;
				}
				if(epoch > 3000)
				{
					flag = false;
				}
			}
			//System.out.println("Total Square Error: " + this.totalSquareError);
			epoch++;
		}
	}

	public void initializeWeightsRandomly()
	{
		Random rand = new Random();
		for(int i = 0; i < this.inputToFirstHiddenLayerWeightsArray.length; i++)
		{
			float randomValue = MIN_RANDOM_VALUE + (MAX_RANDOM_VALUE - MIN_RANDOM_VALUE) * rand.nextFloat();
			this.inputToFirstHiddenLayerWeightsArray[i] = randomValue;
		}
		
		for(int i = 0; i < this.firstToSecondHiddenLayerWeightsArray.length; i++)
		{
			float randomValue = MIN_RANDOM_VALUE + (MAX_RANDOM_VALUE - MIN_RANDOM_VALUE) * rand.nextFloat();
			this.firstToSecondHiddenLayerWeightsArray[i] = randomValue;
		}
		
		for(int i = 0; i < this.secondToOutputLayerWeightsArray.length; i++)
		{
			float randomValue = MIN_RANDOM_VALUE + (MAX_RANDOM_VALUE - MIN_RANDOM_VALUE) * rand.nextFloat();
			this.secondToOutputLayerWeightsArray[i] = randomValue;
		}
		
		for(int i = 0; i < this.firstHiddenLayerBiasArray.length; i++)
		{
			float randomValue = MIN_RANDOM_VALUE + (MAX_RANDOM_VALUE - MIN_RANDOM_VALUE) * rand.nextFloat();
			this.firstHiddenLayerBiasArray[i] = randomValue;
		}
		
		for(int i = 0; i < this.secondHiddenLayerBiasArray.length; i++)
		{
			float randomValue = MIN_RANDOM_VALUE + (MAX_RANDOM_VALUE - MIN_RANDOM_VALUE) * rand.nextFloat();
			this.secondHiddenLayerBiasArray[i] = randomValue;
		}
		
		for(int i = 0; i < this.outputLayerBiasArray.length; i++)
		{
			float randomValue = MIN_RANDOM_VALUE + (MAX_RANDOM_VALUE - MIN_RANDOM_VALUE) * rand.nextFloat();
			this.outputLayerBiasArray[i] = randomValue;
		}
	}
	
	public void initializeTempDerivativeErrors(ArrayList<Float> temp1, ArrayList<Float> temp2, ArrayList<Float> temp3, ArrayList<Float> temp4, ArrayList<Float> temp5, ArrayList<Float> temp6)
	{
		// first hidden layer
		for(int i = 0; i < this.inputToFirstHiddenLayerWeightsArray.length; i++) 
		{
			temp1.add(0.0f);
		}

		// first hidden bias
		for(int i = 0; i < this.firstHiddenLayerBiasArray.length; i++) 
		{
			temp2.add(0.0f);
		}
		
		// second hidden layer
		for(int i = 0; i < this.firstToSecondHiddenLayerWeightsArray.length; i++) 
		{
			temp3.add(0.0f);
		}

		// second hidden bias
		for(int i = 0; i < this.secondHiddenLayerBiasArray.length; i++) 
		{
			temp4.add(0.0f);
		}

		// output layer
		for(int i = 0; i < this.secondToOutputLayerWeightsArray.length; i++) 
		{
			temp5.add(0.0f);
		}

		// output bias
		for(int i = 0; i < this.outputLayerBiasArray.length; i++) 
		{
			temp6.add(0.0f);
		}
	}

	public void setTempDerivativeErrors(ArrayList<Float> temp1, ArrayList<Float> temp2, ArrayList<Float> temp3, ArrayList<Float> temp4, ArrayList<Float> temp5, ArrayList<Float> temp6)
	{
		// first hidden layer
		for(int i = 0; i < this.inputToFirstHiddenLayerWeightsArray.length; i++) 
		{
			temp1.set(i, 0f);
		}

		// first hidden bias
		for(int i = 0; i < this.firstHiddenLayerBiasArray.length; i++) 
		{
			temp2.set(i, 0f);
		}
		
		// second hidden layer
		for(int i = 0; i < this.firstToSecondHiddenLayerWeightsArray.length; i++) 
		{
			temp3.set(i, 0f);
		}

		// second hidden bias
		for(int i = 0; i < this.secondHiddenLayerBiasArray.length; i++) 
		{
			temp4.set(i, 0f);
		}

		// output layer
		for(int i = 0; i < this.secondToOutputLayerWeightsArray.length; i++) 
		{
			temp5.set(i, 0f);
		}

		// output bias
		for(int i = 0; i < this.outputLayerBiasArray.length; i++) 
		{
			temp6.set(i, 0f);
		}
	}
	
	public void initializeDerivativeErrorsWithZeros()
	{
		this.derivativeErrorInputToFirst = new ArrayList<Float>();
		for(int i = 0; i < this.inputToFirstHiddenLayerWeightsArray.length; i++)
		{
			this.derivativeErrorInputToFirst.add(0.0f);
		}
		
		this.derivativeErrorFirstToSecond = new ArrayList<Float>();
		for(int i = 0; i < this.firstToSecondHiddenLayerWeightsArray.length; i++)
		{
			this.derivativeErrorFirstToSecond.add(0.0f);
		}
		
		this.derivativeErrorSecondToOutput = new ArrayList<Float>();
		for(int i = 0; i < this.secondToOutputLayerWeightsArray.length; i++)
		{
			this.derivativeErrorSecondToOutput.add(0.0f);
		}
		
		this.derivativeErrorFirstLayerBias = new ArrayList<Float>();
		for(int i = 0; i < this.firstHiddenLayerBiasArray.length; i++)
		{
			this.derivativeErrorFirstLayerBias.add(0.0f);
		}
		
		this.derivativeErrorSecondLayerBias = new ArrayList<Float>();
		for(int i = 0; i < this.secondHiddenLayerBiasArray.length; i++)
		{
			this.derivativeErrorSecondLayerBias.add(0.0f);
		}
		
		this.derivativeErrorOutputLayerBias = new ArrayList<Float>();
		for(int i = 0; i < this.outputLayerBiasArray.length; i++)
		{
			this.derivativeErrorOutputLayerBias.add(0.0f);
		}
	}
	
	public void updateDerivativeErrors(ArrayList<Float> temp1, ArrayList<Float> temp2, ArrayList<Float> temp3, ArrayList<Float> temp4, ArrayList<Float> temp5, ArrayList<Float> temp6)
	{
		// first hidden layer
		for(int i = 0; i < this.derivativeErrorInputToFirst.size(); i++) 
		{
			temp1.set(i, temp1.get(i) + this.derivativeErrorInputToFirst.get(i));  
		}
		
		// first hidden bias
		for(int i = 0; i < this.derivativeErrorFirstLayerBias.size(); i++) 
		{
			temp2.set(i, temp2.get(i) + this.derivativeErrorFirstLayerBias.get(i));
		}
		
		// second hidden layer
		for(int i = 0; i < this.derivativeErrorFirstToSecond.size(); i++) 
		{
			temp3.set(i, temp3.get(i) + this.derivativeErrorFirstToSecond.get(i));
		}
		
		// second hidden bias
		for(int i = 0; i < this.derivativeErrorSecondLayerBias.size(); i++) 
		{
			temp4.set(i, temp4.get(i) + this.derivativeErrorSecondLayerBias.get(i));
		}
		
		// output layer
		for(int i = 0; i < this.derivativeErrorSecondToOutput.size(); i++) 
		{
			temp5.set(i, temp5.get(i) + this.derivativeErrorSecondToOutput.get(i));
		}
		
		// output bias
		for(int i = 0; i < this.derivativeErrorOutputLayerBias.size(); i++) 
		{
			temp6.set(i, temp6.get(i) + this.derivativeErrorOutputLayerBias.get(i));
		}
	}
	
	public void updateWeights(float learningRate, ArrayList<Float> temp1, ArrayList<Float> temp2, ArrayList<Float> temp3, ArrayList<Float> temp4, ArrayList<Float> temp5, ArrayList<Float> temp6)
	{
		for(int i = 0; i < this.inputToFirstHiddenLayerWeightsArray.length; i++)
		{
			this.inputToFirstHiddenLayerWeightsArray[i] = this.inputToFirstHiddenLayerWeightsArray[i] - (learningRate * temp1.get(i));
		}
		
		for(int i = 0; i < this.firstHiddenLayerBiasArray.length; i++)
		{
			this.firstHiddenLayerBiasArray[i] = this.firstHiddenLayerBiasArray[i] - (learningRate * temp2.get(i));
		}
		
		for(int i = 0; i < this.firstToSecondHiddenLayerWeightsArray.length; i++)
		{
			this.firstToSecondHiddenLayerWeightsArray[i] = this.firstToSecondHiddenLayerWeightsArray[i] - (learningRate * temp3.get(i));
		}
		
		for(int i = 0; i < this.secondHiddenLayerBiasArray.length; i++)
		{
			this.secondHiddenLayerBiasArray[i] = this.secondHiddenLayerBiasArray[i] - (learningRate * temp4.get(i));
		}
		
		for(int i = 0; i < this.secondToOutputLayerWeightsArray.length; i++)
		{
			this.secondToOutputLayerWeightsArray[i] = this.secondToOutputLayerWeightsArray[i] - (learningRate * temp5.get(i));
		}
		
		for(int i = 0; i < this.outputLayerBiasArray.length; i++)
		{
			this.outputLayerBiasArray[i] = this.outputLayerBiasArray[i] - (learningRate * temp6.get(i));
		}
	}
	
	public void backPropagation(float[] example, float[] targetOutput)
	{
		forwardPass(example, targetOutput);
		
		computeOutputLayerDelta(targetOutput);
		
		computeSecondHiddenLayerDelta();
		
		computeFirstHiddenLayerDelta();
		
		computeDerivativeErrors(example);
	}
	
	public void forwardPass(float[] example, float[] targetOutput)
	{
		int weightCounter1 = 0;
		for(int neuron = 0; neuron < this.numberOfFirstHiddenLayerNeurons; neuron++)
		{
			float sum = 0;
			for(int i = 0; i < this.inputDimension; i++)
			{
				sum = sum + (example[i] * this.inputToFirstHiddenLayerWeightsArray[weightCounter1]);
				weightCounter1++;
			}
			sum = sum + this.firstHiddenLayerBiasArray[neuron];
			this.firstHiddenLayerOutput[neuron] = logisticFunction(sum);
		}

		int weightCounter2 = 0;
		for(int neuron = 0; neuron < this.numberOfSecondHiddenLayerNeurons; neuron++)
		{
			float sum = 0;
			for(int i = 0; i < this.numberOfFirstHiddenLayerNeurons; i++)
			{
				sum = sum + (this.firstHiddenLayerOutput[i] * this.firstToSecondHiddenLayerWeightsArray[weightCounter2]);
				weightCounter2++;
			}
			sum = sum + this.secondHiddenLayerBiasArray[neuron];
			if(this.secondHiddenLayerActivationFunction.equals("linear"))
			{
				this.secondHiddenLayerOutput[neuron] = linearFunction(sum);
			}
			else
			{
				this.secondHiddenLayerOutput[neuron] = hyperbolicFunction(sum);
			}
		}

		int weightCounter3 = 0;
		for(int neuron = 0; neuron < this.numberOfOutputLayerNeurons; neuron++)
		{
			float sum = 0;
			for(int i = 0; i < this.numberOfSecondHiddenLayerNeurons; i++)
			{
				sum = sum + (this.secondHiddenLayerOutput[i] * secondToOutputLayerWeightsArray[weightCounter3]);
				weightCounter3++;
			}
			sum = sum + this.outputLayerBiasArray[neuron];
			this.neuralNetworkOutput[neuron] = logisticFunction(sum);
		}
		
		float exampleError = 0.0f;
		for(int i = 0; i < targetOutput.length; i++)
		{
			exampleError = (float) (exampleError + Math.pow((targetOutput[i] - this.neuralNetworkOutput[i]), 2));
		}
		this.totalSquareError = this.totalSquareError + (exampleError / 2);
	}
	
	public void computeOutputLayerDelta(float[] targetOutput)
	{
		for(int neuron = 0; neuron < this.numberOfOutputLayerNeurons; neuron++)
		{
			this.outputLayerDelta[neuron] = this.neuralNetworkOutput[neuron] * (1 - this.neuralNetworkOutput[neuron]) * (this.neuralNetworkOutput[neuron] - targetOutput[neuron]);
		}
	}
	
	public void computeSecondHiddenLayerDelta()
	{
		int neuronIndex = 0;
		for(int neuron = 0; neuron < this.numberOfSecondHiddenLayerNeurons; neuron++)
		{
			float sum = 0;
			int deltaCounter = 0;
			int index = neuronIndex;
			for(int i = 0; i < this.numberOfOutputLayerNeurons; i++)
			{
				sum = sum + this.secondToOutputLayerWeightsArray[index] * this.outputLayerDelta[deltaCounter];
				deltaCounter++;
				index = index + this.numberOfSecondHiddenLayerNeurons;
			}
			if(this.secondHiddenLayerActivationFunction.equals("hyperbolic"))
			{
				this.secondHiddenLayerDelta[neuron] = (float) ( (1 / Math.pow(Math.cosh(this.secondHiddenLayerOutput[neuron]), 2)) * sum );
			}
			else
			{
				this.secondHiddenLayerDelta[neuron] = sum;
			}
			neuronIndex++;
		}
	}
	
	public void computeFirstHiddenLayerDelta()
	{
		int neuronIndex = 0;
		for(int neuron = 0; neuron < this.numberOfFirstHiddenLayerNeurons; neuron++)
		{
			float sum = 0;
			int deltaCounter = 0;
			int index = neuronIndex;
			for(int i = 0; i < this.numberOfSecondHiddenLayerNeurons; i++)
			{
				sum = sum + this.firstToSecondHiddenLayerWeightsArray[index] * this.secondHiddenLayerDelta[deltaCounter];
				deltaCounter++;
				index = index + this.numberOfFirstHiddenLayerNeurons;
			}
			this.firstHiddenLayerDelta[neuron] = this.firstHiddenLayerOutput[neuron] * (1 - this.firstHiddenLayerOutput[neuron]) * sum;
			neuronIndex++;
		}
	}
	
	public void computeDerivativeErrors(float[] example)
	{
		// first hidden layer
		int counter1 = 0;
		for(int i = 0; i < this.firstHiddenLayerDelta.length; i++) 
		{
			for(int j = 0; j < example.length; j++)
			{
				this.derivativeErrorInputToFirst.set(counter1, this.firstHiddenLayerDelta[i]*example[j]);
				counter1++;
			}
		}

		// first hidden bias
		for(int i = 0; i < this.firstHiddenLayerDelta.length; i++) 
		{
			this.derivativeErrorFirstLayerBias.set(i, this.firstHiddenLayerDelta[i]);
		}
		
		// second hidden layer
		int counter3 = 0;
		for(int i = 0; i < this.secondHiddenLayerDelta.length; i++) 
		{
			for(int j = 0; j < this.firstHiddenLayerOutput.length; j++)
			{
				this.derivativeErrorFirstToSecond.set(counter3, this.secondHiddenLayerDelta[i] * this.firstHiddenLayerOutput[j]);
				counter3++;
			}
		}

		// second hidden bias
		for(int i = 0; i < this.secondHiddenLayerDelta.length; i++) 
		{
			this.derivativeErrorSecondLayerBias.set(i, this.secondHiddenLayerDelta[i]);
		}

		// output layer
		int counter5 = 0;
		for(int i = 0; i < this.outputLayerDelta.length; i++) 
		{
			for(int j = 0; j < this.secondHiddenLayerOutput.length; j++)
			{
				this.derivativeErrorSecondToOutput.set(counter5, this.outputLayerDelta[i] * this.secondHiddenLayerOutput[j]);
				counter5++;
			}
		}

		// output bias
		for(int i = 0; i < this.outputLayerDelta.length; i++) 
		{
			this.derivativeErrorOutputLayerBias.set(i, this.outputLayerDelta[i]);
		}
	}
	
	public float logisticFunction(float value)
	{
		return (float)(1f / (1f + Math.exp(-value)));
	}
	
	public float hyperbolicFunction(float value)
	{
		return  (float) Math.tanh(value);
	}
	
	public float linearFunction(float value)
	{
		return value;
	}

	
	public float generalizationEstimation()
	{
		float countCorrectOutputs = 0f;
		for(float[] example: this.testSetHashMap.keySet())
		{
			// example is the key and this.testSetHashMap.get(example) is the value.
			float[] targetOutput = this.testSetHashMap.get(example);
			forwardPass(example, targetOutput);
			
			int maxIndex = 0;
			float maxValue = this.neuralNetworkOutput[0];
			for(int i = 1; i <= 2; i++)
			{
				if(this.neuralNetworkOutput[i] >= maxValue)
				{
					maxValue = this.neuralNetworkOutput[i];
					maxIndex = i;
				}
			}
			
			int maxIndexTarget = 0;
			for(int i = 1; i <= 2; i++)
			{
				if(targetOutput[i] == 1.0f)
				{
					maxIndexTarget = i;
				}
			}

			if(maxIndex == maxIndexTarget)
			{
				countCorrectOutputs++;
				float[] ex = {example[0], example[1], 1};
				this.bestTestSetArrayList.add(ex);
			}
			else
			{
				float[] ex = {example[0], example[1], 0};
				this.bestTestSetArrayList.add(ex);
			}
		}
		//System.out.println("correct: " + countCorrectOutputs);
		//System.out.println("Total: " + this.testSetHashMap.size());
		float generalizationEstimation = (float)((countCorrectOutputs / this.testSetHashMap.size()) * 100);
		System.out.println("Estimation of generalization: " + generalizationEstimation + "%");
		System.out.println("-------------------------------------------------------");
		return generalizationEstimation;
	}
	
	public ArrayList<float[]> getBestTestSet()
	{
		return this.bestTestSetArrayList;
	}

}
