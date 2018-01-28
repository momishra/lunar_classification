
import java.io.IOException;
import java.util.Vector;

import no.uib.cipr.matrix.DenseMatrix;
import no.uib.cipr.matrix.NotConvergedException;

import org.apache.commons.logging.Log;
import org.apache.commons.logging.LogFactory;
import org.apache.hadoop.io.IntWritable;
import org.apache.hadoop.io.LongWritable;
import org.apache.hadoop.io.MapWritable;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.mapred.MapTask;
import org.apache.hadoop.mapreduce.Mapper;
import org.apache.hadoop.mapreduce.filecache.DistributedCache;
import org.apache.log4j.Logger;
import org.apache.zookeeper.KeeperException;

import com.google.gson.Gson;

import Jama.Matrix;

	public class firstMapper extends Mapper <LongWritable,Text, LongWritable, MapWritable> {
		public Matrix train_set;
		public static int numTrainData;
		private int Elm_Type = 1; //0 for regression
		private int NumberofHiddenNeurons;
		private int NumberofOutputNeurons;					
		private int NumberofInputNeurons;						
		private String func = "sig";
		private int []label;
		private DenseMatrix T;
		private DenseMatrix P;
		private DenseMatrix H;
		public static DenseMatrix Z;
		MapWritable mw = new MapWritable();
		 private static final Log LOG = LogFactory.getLog(firstMapper.class.getName());
		
	private Matrix loadMatrix(String data) throws IOException {
			String[] numberOfSamples = data.split("\n");
			String[] numberOfFeatures = numberOfSamples[0].split(",");
            
            int m = numberOfSamples.length;
    		int n = numberOfFeatures.length;
    		
    		Matrix matrix = new Matrix(m, n);
    		int i = 0;
    		while (i<m) {
    			String []datatrings = numberOfSamples[i].split(",");
    			for (int j = 0; j < n; j++) {
    				matrix.set(i, j, Double.parseDouble(datatrings[j]));
    			}
    			i++;
    		}
    		return matrix;
		}
	
	public void run(Context context) throws IOException,InterruptedException {
		setup(context);
		try {
			while (context.nextKeyValue()) {
				map(context.getCurrentKey(), context.getCurrentValue(), context);
			}
		} finally {
			cleanup(context);
		}
	}
	
	protected void setup(Context context) throws IOException,InterruptedException {
		NumberofHiddenNeurons = context.getConfiguration().getInt("NumberofHiddenNeurons", 0);
		NumberofOutputNeurons =  context.getConfiguration().getInt("NumberofOutputNeurons", 0);
	} 
		
	protected void map(LongWritable key, Text value, Context context) throws IOException, InterruptedException {
		String data = value.toString();
		train_set = loadMatrix(data);
		try {
			train();
		} catch (NotConvergedException e) {
			e.printStackTrace();
		}
		mw.put(new IntWritable(0), DoubleArrayWritable.getWritable(H));
		mw.put(new IntWritable(1), DoubleArrayWritable.getWritable(T));
		context.write(key, mw);
	}

	private void train() throws NotConvergedException {
		LOG.info("Number of Hidden nuerons :"+ NumberofHiddenNeurons);
		LOG.info("Number of Output nuerons :" + NumberofOutputNeurons);
			numTrainData = train_set.getRowDimension();
			NumberofInputNeurons = train_set.getColumnDimension()-2;
			DenseMatrix transT = new DenseMatrix(1, numTrainData);
			DenseMatrix transP = new DenseMatrix(numTrainData, NumberofInputNeurons);
			for (int i = 0; i < numTrainData; i++) {
				transT.set(0, i, train_set.get(i, 0));
				for (int j = 1; j <= NumberofInputNeurons; j++)
					transP.set(i, j-1, train_set.get(i, j+1));
			}
			T = new DenseMatrix(numTrainData,1);
			P = new DenseMatrix(NumberofInputNeurons,numTrainData);
			transT.transpose(T);
			transP.transpose(P);
			
			if(Elm_Type != 0)	//CLASSIFIER
			{
				label = new int[NumberofOutputNeurons];
				for (int i = 0; i < NumberofOutputNeurons; i++) {
					label[i] = i+1;						
				}
				DenseMatrix tempT = new DenseMatrix(NumberofOutputNeurons,numTrainData);
				tempT.zero();
				for (int i = 0; i < numTrainData; i++){
						int j = 0;
				        for (j = 0; j < NumberofOutputNeurons; j++){
				            if (label[j] == T.get(i, 0))
				                break; 
				        }
				        tempT.set(j, i, 1); 
				}
				
				transT = new DenseMatrix(NumberofOutputNeurons,numTrainData);
				for (int i = 0; i < NumberofOutputNeurons; i++){
			        for (int j = 0; j < numTrainData; j++)
			        	transT.set(i, j, tempT.get(i, j)*2-1);
				}
				
				T = new DenseMatrix(numTrainData,NumberofOutputNeurons);
				transT.transpose(T);
			} 	//end if CLASSIFIER
			
			
			///////////////////////////////////////////////////////////////////////////////////
			
			ZKClientManagerImpl zk = new ZKClientManagerImpl();
			Gson gson = new Gson();
			try {
				String inputStr = (String) zk.getZNodeData("/inputWeight", false);//can't cast it!!
				String biasStr = (String) zk.getZNodeData("/bias", false);
				DenseMatrix input = gson.fromJson(inputStr, DenseMatrix.class);
				DenseMatrix bias = gson.fromJson(biasStr, DenseMatrix.class);
				calculateParameters(input, bias);
			} catch (KeeperException e) {
				// TODO Auto-generated catch block
				e.printStackTrace();
			} catch (InterruptedException e) {
				// TODO Auto-generated catch block
				e.printStackTrace();
			}
			////////////////////////////////////////////////////////////////////////////////////
		}

		public void calculateParameters(DenseMatrix InputWeight, DenseMatrix BiasOfHiddenNeuron) throws NotConvergedException{
			
			DenseMatrix tempH = new DenseMatrix(NumberofHiddenNeurons, numTrainData); 
			InputWeight.mult(P, tempH);
			
			DenseMatrix BiasMatrix = new DenseMatrix(NumberofHiddenNeurons, numTrainData);
			
			for (int j = 0; j < numTrainData; j++) {
				for (int i = 0; i < NumberofHiddenNeurons; i++) {
					BiasMatrix.set(i, j, BiasOfHiddenNeuron.get(i, 0)*2-1);
				}
			}
		
			tempH.add(BiasMatrix);
			DenseMatrix Ht = new DenseMatrix(NumberofHiddenNeurons, numTrainData);
			
			if(func.startsWith("sig")){
				for (int j = 0; j < NumberofHiddenNeurons; j++) {
					for (int i = 0; i < numTrainData; i++) {
						double temp = tempH.get(j, i);
						temp = 1.0f/ (1 + Math.exp(-temp));
						Ht.set(j, i, temp);
					}
				}
			}
			else if(func.startsWith("sin")){
				for (int j = 0; j < NumberofHiddenNeurons; j++) {
					for (int i = 0; i < numTrainData; i++) {
						double temp = tempH.get(j, i);
						temp = Math.sin(temp);
						Ht.set(j, i, temp);
					}
				}
			}
			H = new DenseMatrix(numTrainData,NumberofHiddenNeurons);
			Ht.transpose(H);
		}
	}