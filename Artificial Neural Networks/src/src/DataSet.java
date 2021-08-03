package src;

import java.io.FileWriter;
import java.io.IOException;
import java.util.Random;

public class DataSet 
{
	private static final float MIN = -2;
	private static final float MAX = 2;
	private float[] array = new float[2];

	public void  createTrainingSet()
	{
		try
		{
			FileWriter writer = new FileWriter("TrainingSet.txt");
			for(int i = 0; i < 3000; i++)
			{
				this.array = createRandomNumbers();
				int categoryNumber = checkCategory(array[0], array[1]);
				int newCategoryNumber = categoryNumber;
				if(categoryNumber != 1)
				{
					newCategoryNumber = putNoiseOnTrainingSet(categoryNumber);
				}
				String line = this.array[0] + "," + this.array[1] + "," + newCategoryNumber + "\n";
				writer.write(line);
			}
			writer.close();
		}
		catch(IOException e)
		{
			e.printStackTrace();
		}
	}
	
	public void  createTestSet()
	{
		try
		{
			FileWriter writer = new FileWriter("TestSet.txt");
			for(int i = 0; i < 3000; i++)
			{
				this.array = createRandomNumbers();
				int categoryNumber = checkCategory(this.array[0], this.array[1]);
				String line = this.array[0] + "," + this.array[1] + "," + categoryNumber + "\n";
				writer.write(line);
			}
			writer.close();
		}
		catch(IOException e)
		{
			e.printStackTrace();
		}
	}
		
	public float[]  createRandomNumbers()
	{
		float x1 = (float) (MIN + (Math.random() * ((MAX - MIN))));
		float x2 = (float) (MIN + (Math.random() * ((MAX - MIN))));
		
		float[] tempArray = new float[2];;
		tempArray[0] = x1;
		tempArray[1] = x2;
		
		return tempArray;
	}
	
	public int checkCategory(float x1, float x2)
	{
		if(checkSecondCategory(x1, x2))
		{
			return 2;
		}
		else if (checkThirdCategory(x1, x2))
		{
			return 3;
		}
		else
		{
			return 1;
		}
	}
	
	public boolean checkSecondCategory(float x1, float x2)
	{
		if(((Math.pow((x1-1), 2) + Math.pow((x2-1), 2)) <= 0.49) || ((Math.pow((x1+1), 2) + Math.pow((x2+1), 2)) <= 0.49))
		{
			return true;
		}
		return false;
	}
	
	public boolean checkThirdCategory(float x1, float x2)
	{
		if(((Math.pow((x1+1), 2) + Math.pow((x2-1), 2)) <= 0.49) || ((Math.pow((x1-1), 2) + Math.pow((x2+1), 2)) <= 0.49))
		{
			return true;
		}
		return false;
	}
	
	public int putNoiseOnTrainingSet(int categoryNumber)
	{
		float[] propability = {0.0f, 0.1f, 0.2f, 0.3f, 0.4f, 0.5f, 0.6f, 0.7f, 0.8f, 0.9f};
		
		int randomIndex = new Random().nextInt(propability.length);
		if(randomIndex == 1)
		{
			return 1;
		}
		else
		{
			return categoryNumber;
		}
	}

}
