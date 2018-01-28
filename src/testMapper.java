import java.io.IOException;
import java.net.URI;

import no.uib.cipr.matrix.DenseMatrix;
import no.uib.cipr.matrix.NotConvergedException;

import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.DoubleWritable;
import org.apache.hadoop.io.IntWritable;
import org.apache.hadoop.io.LongWritable;
import org.apache.hadoop.io.MapWritable;
import org.apache.hadoop.io.SequenceFile;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.mapreduce.Mapper;
import org.apache.hadoop.mapreduce.filecache.DistributedCache;
import org.apache.zookeeper.KeeperException;

import com.google.gson.Gson;

	public class testMapper extends Mapper<LongWritable,Text, LongWritable, DoubleWritable> {
		private double TrainingAccuracy;
		private int []label;
		private int numTestData;
		private DenseMatrix test_set;
		private DenseMatrix Y;
		private DenseMatrix T;
		private DenseMatrix P;
		private DenseMatrix H;
		SequenceFile.Reader reader;
		DenseMatrix Z;
		private int NumberofHiddenNeurons;
		private int NumberofOutputNeurons;
		private int NumberofInputNeurons;
		
		protected void setup(Context context) throws IOException,InterruptedException {
			NumberofHiddenNeurons = context.getConfiguration().getInt("NumberofHiddenNeurons", 0);
			NumberofOutputNeurons =  context.getConfiguration().getInt("NumberofOutputNeurons", 0);
			
			FileSystem fs = FileSystem.get(context.getConfiguration());
			URI files[] = DistributedCache.getCacheFiles(context.getConfiguration());
			Path path = new Path(files[0].toString());
			reader = new SequenceFile.Reader(fs, path, context.getConfiguration());
		}
		
	private DenseMatrix loadMatrix(String data) throws IOException {
			String[] numberOfSamples = data.split("\n");
			String[] numberOfFeatures = numberOfSamples[0].split(",");
            
            int m = numberOfSamples.length;
    		int n = numberOfFeatures.length;
    		
    		DenseMatrix matrix = new DenseMatrix(m, n);
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
		
	protected void map(LongWritable key, Text value, Context context) throws IOException, InterruptedException {
		String data = value.toString();
		test_set = loadMatrix(data);
		//load Z here
		try {
			LongWritable keys = (LongWritable) reader.getKeyClass().newInstance();
			MapWritable values = (MapWritable) reader.getValueClass().newInstance();
			while(reader.next(keys,values)){
					Z = new DenseMatrix(NumberofHiddenNeurons,NumberofOutputNeurons);
					// fetch value of H
					for (int r = 0; r < Z.numRows(); r++) {
						for (int c = 0; c < Z.numColumns(); c++) {
							Z.set(r, c, Double.parseDouble(((((DoubleArrayWritable) values.get(new IntWritable(0))).get()[r][c]).toString())));
						}
				}
				break;
			}
		} catch (InstantiationException e1) {
			// TODO Auto-generated catch block
			e1.printStackTrace();
		} catch (IllegalAccessException e1) {
			// TODO Auto-generated catch block
			e1.printStackTrace();
		}
		
		try {
			test();
		} catch (NotConvergedException e) {
			e.printStackTrace();
		}
		context.write(key,new DoubleWritable(TrainingAccuracy));
		/////Here, TrainingAccuracy is printed, is it correct?///
	}

	private void test() throws NotConvergedException {
		numTestData = test_set.numRows();
		NumberofInputNeurons = test_set.numColumns() -2;
		DenseMatrix transT = new DenseMatrix(1, numTestData);
		DenseMatrix transP = new DenseMatrix(numTestData,NumberofInputNeurons);
		for (int i = 0; i < numTestData; i++) {
			transT.set(0, i, test_set.get(i, 0));
			for (int j = 1; j <= NumberofInputNeurons; j++)
				transP.set(i, j - 1, test_set.get(i, j + 1));
		}
		T = new DenseMatrix(numTestData, 1);
		P = new DenseMatrix(NumberofInputNeurons, numTestData);
		transT.transpose(T);
		transP.transpose(P);

		label = new int[NumberofOutputNeurons];
		for (int i = 0; i < label.length; i++) {
			label[i] = i + 1;
		}
		DenseMatrix tempT = new DenseMatrix(NumberofOutputNeurons,numTestData);
		tempT.zero();
		for (int i = 0; i < numTestData; i++) {
			int j = 0;
			for (j = 0; j < NumberofOutputNeurons; j++) {
				if (label[j] == T.get(i, 0))
					break;
			}
			tempT.set(j, i, 1);
		}

		transT = new DenseMatrix(NumberofOutputNeurons, numTestData);
		for (int i = 0; i < NumberofOutputNeurons; i++) {
			for (int j = 0; j < numTestData; j++)
				transT.set(i, j, tempT.get(i, j) * 2 - 1);
		}

		T = new DenseMatrix(numTestData, NumberofOutputNeurons);
		transT.transpose(T);
		ZKClientManagerImpl zk = new ZKClientManagerImpl();
		Gson gson = new Gson();
		try {
			String inputStr = (String) zk.getZNodeData("/inputWeight", false);//can't cast it!!
			String biasStr = (String) zk.getZNodeData("/bias", false);
			DenseMatrix input = gson.fromJson(inputStr, DenseMatrix.class);
			DenseMatrix bias = gson.fromJson(biasStr, DenseMatrix.class);
			calculateOutputWeight(input, bias);
		} catch (KeeperException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		} catch (InterruptedException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
		
	}

	public void calculateOutputWeight(DenseMatrix InputWeight,DenseMatrix BiasOfHiddenNeuron) throws NotConvergedException {

		DenseMatrix tempH = new DenseMatrix(NumberofHiddenNeurons,numTestData);
		InputWeight.mult(P, tempH);

		DenseMatrix BiasMatrix = new DenseMatrix(NumberofHiddenNeurons, numTestData);

		for (int j = 0; j < numTestData; j++) {
			for (int i = 0; i < NumberofHiddenNeurons; i++) {
				BiasMatrix.set(i, j,BiasOfHiddenNeuron.get(i, 0) * 2 - 1);
			}
		}

		tempH.add(BiasMatrix);
		DenseMatrix Ht = new DenseMatrix(NumberofHiddenNeurons,
				numTestData);

		for (int j = 0; j < NumberofHiddenNeurons; j++) {
			for (int i = 0; i < numTestData; i++) {
				double temp = tempH.get(j, i);
				temp = 1.0f / (1 + Math.exp(-temp));
				Ht.set(j, i, temp);
			}
		}
		H = new DenseMatrix(numTestData, NumberofHiddenNeurons);
		Ht.transpose(H);
		calculateTestingAccuracy();
	}				
		
		public double calculateTestingAccuracy() {
			DenseMatrix Yt = new DenseMatrix(numTestData,NumberofOutputNeurons);
			H.mult(Z,Yt);
			Y = new DenseMatrix(NumberofOutputNeurons,numTestData);
			Yt.transpose(Y);
			float MissClassificationRate_Training=0.0f;
				
			    for (int i = 0; i < numTestData; i++) {
					double maxtag1 = Y.get(0, i);
					int tag1 = 0;
					double maxtag2 = T.get(i, 0);
					int tag2 = 0;
			    	for (int j = 1; j < NumberofOutputNeurons; j++) {
						if(Y.get(j, i) > maxtag1){
							maxtag1 = Y.get(j, i);
							tag1 = j;
						}
						if(T.get(i, j) > maxtag2){
							maxtag2 = T.get(i, j);
							tag2 = j;
						}
					}
			    	if(tag1 != tag2)
			    		MissClassificationRate_Training ++;
				}
			   TrainingAccuracy = 1 - MissClassificationRate_Training*1.0f/numTestData;	
			return TrainingAccuracy;
		}
	 
	}