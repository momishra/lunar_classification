import java.io.IOException;
import java.net.URI;
import no.uib.cipr.matrix.DenseMatrix;
import no.uib.cipr.matrix.Matrices;
import no.uib.cipr.matrix.NotConvergedException;

import org.apache.commons.logging.Log;
import org.apache.commons.logging.LogFactory;
import org.apache.hadoop.fs.FSDataInputStream;
import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.IntWritable;
import org.apache.hadoop.io.LongWritable;
import org.apache.hadoop.io.MapWritable;
import org.apache.hadoop.io.SequenceFile;
import org.apache.hadoop.mapreduce.Mapper;
import org.apache.hadoop.mapreduce.filecache.DistributedCache;
import org.apache.hadoop.util.Time;
/* 
 * @author Mohini Mishra 
 */

public class secMapper extends Mapper<LongWritable, MapWritable, LongWritable, MapWritable>{
	private double rho;
	static double converT =1.6;
	static double C1 = 10;
	private static DenseMatrix U;
	private DenseMatrix H;
	private DenseMatrix T;
	private static DenseMatrix OutputWeight;
	private int NumberofHiddenNeurons;
	private int NumberofOutputNeurons;	
	private DenseMatrix Z;
	FSDataInputStream in;
	SequenceFile.Reader reader;
	private int iteration;
	private Path path;
	MapWritable m2w = new MapWritable();
	static int MapCalled;
	private int numTrainData=18000; //have to remove this hard-coded value here
    private static final Log LOG = LogFactory.getLog(secMapper.class.getName());
	 
	public void run(Context context) throws IOException, InterruptedException{
		try {
			while (context.nextKeyValue()) {
				setup(context);
				map(context.getCurrentKey(),context.getCurrentValue(),context);
			}
		} finally{
			cleanup(context);
		}
	}
	
	protected void setup(Context context) throws IOException {
	rho = context.getConfiguration().getDouble("rho",0.0);
	iteration = context.getConfiguration().getInt("iterCount", 0);
	NumberofHiddenNeurons = context.getConfiguration().getInt("NumberofHiddenNeurons", 0);
	NumberofOutputNeurons =  context.getConfiguration().getInt("NumberofOutputNeurons", 0);
	
	//////////////////////////////////////////////////////////
	LOG.info("NumberofHiddenNeurons "+NumberofHiddenNeurons);
	LOG.info("Jar is updated :"+ Time.now());
	LOG.info("NumberofOutputNeurons "+NumberofOutputNeurons);
	LOG.info("Number of train data "+numTrainData);
	URI files[] = DistributedCache.getCacheFiles(context.getConfiguration());
	if(files!=null){
		LOG.info("DistributedCache number of Files "+files.length);
		path = new Path(files[0].toString());
	}
	////////////How do I get LOG.info file?////////////////////////////
} 
	protected void map(LongWritable key, MapWritable value, Context context) throws IOException, InterruptedException {
		LOG.info("Inside secMapper.map() function");
		// For reading H and T values from the cached file
		FileSystem fs = FileSystem.get(context.getConfiguration());
		reader = new SequenceFile.Reader(fs, path, context.getConfiguration());
		
		///////////////////////////What is the purpose of the creation newInstance()//////////
		
		try {
			LongWritable keys = (LongWritable) reader.getKeyClass().newInstance();
			MapWritable values = (MapWritable) reader.getValueClass().newInstance();
			while (reader.next(keys)) {
				if (keys.equals(key)) {
					H = new DenseMatrix(numTrainData,NumberofHiddenNeurons);
					T = new DenseMatrix(numTrainData,NumberofOutputNeurons);
					reader.getCurrentValue(values);
					for (int r = 0; r < H.numRows(); r++) {
						for (int c = 0; c < H.numColumns(); c++) {
							H.set(r,c,Double.parseDouble(((((DoubleArrayWritable) values.get(new IntWritable(0))).get()[r][c]).toString())));
						}

						for (int c = 0; c < T.numColumns(); c++) {
							T.set(r,c,Double.parseDouble(((((DoubleArrayWritable) values.get(new IntWritable(1))).get()[r][c]).toString())));
						}
					}
					break;
				}
			}
		} catch (InstantiationException e1) {
			e1.printStackTrace();
		} catch (IllegalAccessException e1) {
			e1.printStackTrace();
		}

		Z = new DenseMatrix(NumberofHiddenNeurons, NumberofOutputNeurons);
		U = new DenseMatrix(NumberofHiddenNeurons, NumberofOutputNeurons);
		if (iteration == 0) {
			U.zero();
			Z.zero();
		} else {
			// reading Z and U map from the output
			if (value.containsKey(key)) {
				for (int r = 0; r < Z.numRows(); r++) {
					for (int c = 0; c < Z.numColumns(); c++) {
						Z.set(r, c, Double.parseDouble(((((DoubleArrayWritable) value.get(new IntWritable(0))).get()[r][c]).toString())));
						U.set(r, c, Double.parseDouble(((((DoubleArrayWritable) value.get(key)).get()[r][c]).toString())));
					}
				}
			}
			
			//updating value of u here only??
		}

		try {
			OutputWeight = calculateOutputWeight(H, T);
		} catch (NotConvergedException e) {
			e.printStackTrace();
		}

		calculateTrainingAccuracy();
		m2w.put(new IntWritable(0), DoubleArrayWritable.getWritable(OutputWeight));
		m2w.put(new IntWritable(1), DoubleArrayWritable.getWritable(Z));
		m2w.put(new IntWritable(2), DoubleArrayWritable.getWritable(U));
		m2w.put(new IntWritable(3), key);
		LOG.info("About to Write to map context in secMapper");
		context.write(new LongWritable(iteration), m2w);
	}

	private DenseMatrix calculateOutputWeight(DenseMatrix h, DenseMatrix T) throws NotConvergedException {
		DenseMatrix ht = new DenseMatrix(h.numColumns(),h.numRows());
		h.transpose(ht);
		DenseMatrix hth = new DenseMatrix(ht.numRows(),ht.numRows());
		ht.mult(h, hth);  
		DenseMatrix identity = Matrices.identity(hth.numRows());
		identity.scale(rho/C1);
		hth.add(identity);
		Inverse invers = new Inverse(hth); 
		DenseMatrix pinvht = invers.getMPInverse(0.000001);			
		DenseMatrix htT = new DenseMatrix(ht.numRows(), T.numColumns());
		ht.mult(T, htT);
		DenseMatrix constPart = new DenseMatrix(pinvht.numRows(), htT.numColumns());
		pinvht.mult(htT,constPart);
		DenseMatrix diffZU = new DenseMatrix(Z.numRows(), Z.numColumns());
		for (int r = 0; r < diffZU.numRows(); r++) {
			for (int c = 0; c < diffZU.numColumns(); c++) {
				diffZU.set(r, c, Z.get(r, c) - U.get(r, c));
			}
		}
		diffZU.scale(rho / C1);
		DenseMatrix tempOutputWeight = new DenseMatrix(Z.numRows(), Z.numColumns());
		pinvht.mult(diffZU, tempOutputWeight);
		tempOutputWeight.add(constPart);
		tempOutputWeight.scale(converT);
		DenseMatrix scaledZ = Z.copy();
		scaledZ.scale(1-converT);
		tempOutputWeight.add(scaledZ);
		return tempOutputWeight;
	}
	
	public void calculateTrainingAccuracy() throws IOException {
		DenseMatrix Yt = new DenseMatrix(numTrainData,NumberofOutputNeurons);
		H.mult(Z,Yt);
		DenseMatrix Y = new DenseMatrix(NumberofOutputNeurons,numTrainData);
		Yt.transpose(Y);
		float MissClassificationRate_Training=0.0f;
			
		    for (int i = 0; i < numTrainData; i++) {
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
		   double TrainingAccuracy = 1 - MissClassificationRate_Training*1.0f/numTrainData;	
		   System.out.println("Training Accuracy :"+ TrainingAccuracy);
	}
}
