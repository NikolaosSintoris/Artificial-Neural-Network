package src;

import java.util.ArrayList;
import java.util.Scanner;

import java.awt.Color;  
import javax.swing.JFrame;  
import javax.swing.SwingUtilities;  
import javax.swing.WindowConstants;  
import org.jfree.chart.ChartFactory;  
import org.jfree.chart.ChartPanel;  
import org.jfree.chart.JFreeChart;  
import org.jfree.chart.plot.XYPlot;  
import org.jfree.data.xy.XYDataset;  
import org.jfree.data.xy.XYSeries;  
import org.jfree.data.xy.XYSeriesCollection;

public class MLPMain extends JFrame
{
	private static final long serialVersionUID = 6294689542092367723L; 
	private static float bestGeneralization;
	private static ArrayList<float[]> bestTestArrayList;
	
	public MLPMain(String title, int inputDimension, int numberOfCategories)
	{
		super(title);
		
		bestGeneralization = 0f;
		
		bestTestArrayList = new ArrayList<float[]>();
		
	    XYDataset dataset = createDataset(inputDimension, numberOfCategories);  

	    // Create chart. 
	    JFreeChart chart = ChartFactory.createScatterPlot("MLP Scatter Plot", "X-Axis", "Y-Axis", dataset); 
	    
	    //Changes background color. 
	    //XYPlot plot = (XYPlot)chart.getPlot();  
	    //plot.setBackgroundPaint(new Color(255,228,196)); 
	    
	    // Create Panel.
	    ChartPanel panel = new ChartPanel(chart);  
	    setContentPane(panel);  
	}
	
	private XYDataset createDataset(int inputDimension, int numberOfCategories)
	{
		XYSeriesCollection dataset = new XYSeriesCollection(); 
		 
		System.out.println("Creation of data set and training set.");
		DataSet data = new DataSet();
		data.createTrainingSet();
		data.createTestSet();
		System.out.println("End of creation\n");
		System.out.println();
		
		//---------------------------------------------------------------------------
		ArrayList<int[]> hiddenNeuronSizeArrayList = new ArrayList<int[]>();
		int[] firstSizeArray = {5, 3};
		int[] secondSizeArray = {7, 4};
		int[] thirdSizeArray = {8, 5};
		hiddenNeuronSizeArrayList.add(firstSizeArray);
		hiddenNeuronSizeArrayList.add(secondSizeArray);
		hiddenNeuronSizeArrayList.add(thirdSizeArray);
		
		String[] activationFunctionArray = {"hyperbolic", "linear"};
		
		int[] miniBatchesSizeArray = {1, 300, 30, 3000};
		int c = 1;
		for(int i = 0; i < activationFunctionArray.length; i++)
		{
			String activationFunction = activationFunctionArray[i];
			
			for(int j = 0; j < hiddenNeuronSizeArrayList.size(); j++)
			{
				int[] currentSize = hiddenNeuronSizeArrayList.get(j);
				
				for(int k = 0; k < miniBatchesSizeArray.length; k++)
				{
					System.out.println("Case: " + c);
					c++;
					int miniBatchesSize = miniBatchesSizeArray[k];
					
					MultiLayerPerceptron mlp = new MultiLayerPerceptron(inputDimension, numberOfCategories, currentSize[0], currentSize[1], activationFunction);
					
					mlp.loadTrainingSet();
					mlp.loadTestSet();

					System.out.println("Ôrain");
					mlp.gradientDescentTraining(miniBatchesSize, 0.001f);
					
					System.out.println("Test");
					float tempGeneralizationEstimation = mlp.generalizationEstimation();
					System.out.println();
					
					if(tempGeneralizationEstimation >= bestGeneralization)
					{
						bestGeneralization = tempGeneralizationEstimation;
						bestTestArrayList = mlp.getBestTestSet();
					}
				}
			}
		}
		System.out.println("Best Generalization: " + bestGeneralization);
		//---------------------------------------------------------------------------
		
		
		XYSeries series1 = new XYSeries("Correct"); 
		XYSeries series2 = new XYSeries("Wrong");
		for(int i = 0; i < bestTestArrayList.size(); i++)
		{
			float[] example = bestTestArrayList.get(i);
			
			if(example[2] == 1)
			{
				series1.add(example[0], example[1]); 
			}
			else
			{
				series2.add(example[0], example[1]);
			}
		}
	    dataset.addSeries(series1);
	    dataset.addSeries(series2); 
		
	    return dataset;  
	}

	public static void main(String[] args) 
	{
		Scanner userInput = new Scanner(System.in);


		System.out.println("Give dimension of examples: ");
		int inputDimension = userInput.nextInt();
		System.out.println();
		
		System.out.println("Give number of categories: ");
		int numberOfCategories = userInput.nextInt();
		System.out.println();

	    SwingUtilities.invokeLater(() -> {  
	    	MLPMain object = new MLPMain("Scatter Plot", inputDimension, numberOfCategories);  
	    	object.setSize(800, 400);  
	    	object.setLocationRelativeTo(null);  
	    	object.setDefaultCloseOperation(WindowConstants.EXIT_ON_CLOSE);  
	    	object.setVisible(true);  
	      }); 

	    userInput.close();
	}  
	
}
