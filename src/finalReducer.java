import java.io.IOException;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.Iterator;
import java.util.Map;

import no.uib.cipr.matrix.DenseMatrix;

import org.apache.commons.logging.Log;
import org.apache.commons.logging.LogFactory;
import org.apache.hadoop.io.IntWritable;
import org.apache.hadoop.io.LongWritable;
import org.apache.hadoop.io.MapWritable;
import org.apache.hadoop.io.SequenceFile;
import org.apache.hadoop.mapred.Counters.Counter;
import org.apache.hadoop.mapred.JobClient;
import org.apache.hadoop.mapred.JobConf;
import org.apache.hadoop.mapred.Reporter;
import org.apache.hadoop.mapreduce.Mapper;
import org.apache.hadoop.mapreduce.ReduceContext;
import org.apache.hadoop.mapreduce.Reducer;
import org.apache.hadoop.mapreduce.filecache.DistributedCache;
import org.apache.http.impl.SocketHttpClientConnection;

import Jama.Matrix;

public class finalReducer  extends Reducer<LongWritable, MapWritable, LongWritable, MapWritable> {
	MapWritable test = new MapWritable();
	SequenceFile.Reader reader;
	private int iteration;
	private int NumberofHiddenNeurons;
	private int NumberofOutputNeurons;	
	DenseMatrix prevZ;
	private double rho;
	private int numMappers;
	private HashMap<LongWritable,DenseMatrix> uMap = new HashMap<LongWritable, DenseMatrix>();
	private HashMap<LongWritable,DenseMatrix> outputWeightMap = new HashMap<LongWritable, DenseMatrix>();
	
	private static final Log LOG = LogFactory.getLog(finalReducer.class.getName());
	//override this method to change
	public void run(Context context) throws IOException, InterruptedException {
		setup(context);
		try {
			 while (context.nextKey()) {
				 reduce(context.getCurrentKey(), context.getValues(), context);
			 }
		}finally{
			cleanup(context);
		}
	}
	
	protected void setup(Context context) throws IOException,InterruptedException {
		JobConf tru = new JobConf(context.getConfiguration());
		numMappers = tru.getNumMapTasks();
		LOG.info("Number of Mappers called :"+ numMappers);
		rho = context.getConfiguration().getDouble("rho",0.0);
		iteration = context.getConfiguration().getInt("iterCount", 0);
		NumberofHiddenNeurons = context.getConfiguration().getInt("NumberofHiddenNeurons", 0);
		NumberofOutputNeurons =  context.getConfiguration().getInt("NumberofOutputNeurons", 0);
	} 
	
	public void reduce(LongWritable key, Iterable<MapWritable> values, Context context) throws IOException, InterruptedException{
		/*the master keeps a clock k, which starts from zero and is incremented by 1 after each z update*/
		/*In the implementation, a counter τ i is kept by the master for each
		worker i. When (x i , u i ) from worker i arrives at the master, the corresponding
		τ i is reset to 1; otherwise, τ i is incremented by 1 as the master’s clock k increments.*/
		
		LOG.info("Reducer called for iteration " + iteration);
		ArrayList<LongWritable> cKeys = new ArrayList<LongWritable>();
		
		for (MapWritable value : values) {
			DenseMatrix matrix1 = new DenseMatrix(NumberofHiddenNeurons, NumberofOutputNeurons);
			DenseMatrix matrix2 = new DenseMatrix(NumberofHiddenNeurons,NumberofOutputNeurons);
			prevZ = new DenseMatrix(NumberofHiddenNeurons,NumberofOutputNeurons);
			for (int r = 0; r < NumberofHiddenNeurons; r++) {
				for (int c = 0; c < NumberofOutputNeurons; c++) {
			    matrix1.set(r,c,Double.parseDouble(((((DoubleArrayWritable) value.get(new IntWritable(0))).get()[r][c]).toString())));
				prevZ.set(r, c, Double.parseDouble(((((DoubleArrayWritable) value.get(new IntWritable(1))).get()[r][c]).toString())));
				matrix2.set(r,c,Double.parseDouble(((((DoubleArrayWritable) value.get(new IntWritable(2))).get()[r][c]).toString())));
				}
			}
			LongWritable splitKey = (LongWritable) value.get(new IntWritable(3));
			outputWeightMap.put(splitKey, matrix1);
			uMap.put(splitKey, matrix2);
			cKeys.add(splitKey);
		}
		
		//finding sum of outputweights and u for calculation of Z
		int i = 0;
		DenseMatrix sum = new DenseMatrix(NumberofHiddenNeurons,NumberofOutputNeurons);
		sum.zero();
		while(i<cKeys.size()){
			LongWritable checkKey = cKeys.get(i); 
			if(outputWeightMap.containsKey(checkKey) && uMap.containsKey(checkKey)){
				sum.add(outputWeightMap.get(checkKey));
				sum.add(uMap.get(checkKey));
			}
			i++;
		}
		
	    //updating z
		double divider = (numMappers+1)/rho;
    	sum.scale(1/divider);
    	DenseMatrix Z = sum;

    	//updating u
    	i=0;
		while(i<cKeys.size()){
			LongWritable checkKey = cKeys.get(i); 
			DenseMatrix temp = Z.copy();
			DenseMatrix avg = new DenseMatrix(NumberofHiddenNeurons,NumberofOutputNeurons);
			avg.zero();
			if(outputWeightMap.containsKey(checkKey) && uMap.containsKey(checkKey)){
				avg.add(outputWeightMap.get(checkKey));
				avg.add(uMap.get(checkKey));
				temp.scale(-1);
				avg.add(temp);
				uMap.put(checkKey,avg);
			}
			i++;
		}
    	
		if (isConverged(Z)) { //checking the convergence criteria
			context.getCounter("iterationCounter","iteration").increment(1L);
		}
        
		for (Map.Entry<LongWritable,DenseMatrix> entry : uMap.entrySet()) {
		    if(null != entry.getKey() && null != entry.getValue()){
		       test.put(entry.getKey(), DoubleArrayWritable.getWritable(entry.getValue()));
		    }
		}
		test.put(new IntWritable(0),DoubleArrayWritable.getWritable(Z));
		for(int m=0;m<cKeys.size();m++){ //doing this so that mappers whose U got updated should only update their corresponding Z value.
			context.write(cKeys.get(m),test); 
		}
	}
	
	private boolean isConverged(DenseMatrix z) {
		boolean result;
		double primal = stoppingCriteria1(z);
		double sec = stoppingcriteria2(z,prevZ);
		result = (primal < 1) || (sec < 0.01) ? true : false;
		System.out.println("Primal residual value :"+ primal +" and Dual residual value "+sec);
		return result;
	}
	
	private double stoppingCriteria1(DenseMatrix z) {
		double norm =0.0;
		Matrix diff = new Matrix (z.numRows(),z.numColumns());
		for (Map.Entry<LongWritable,DenseMatrix> entry : outputWeightMap.entrySet()) {
		    if(null != entry.getKey() && null != entry.getValue()){
		    	for(int r=0;r<diff.getRowDimension();r++){
					for(int c=0;c<diff.getColumnDimension();c++){
						diff.set(r,c,entry.getValue().get(r,c) - z.get(r,c));
					}
				}	
		    }
		    norm += diff.norm2();
		}
		return norm;
	}
	
	private double stoppingcriteria2(DenseMatrix z,DenseMatrix previousz) {
		Matrix diff = new Matrix(z.numRows(),z.numColumns());
		for(int r=0;r<z.numRows();r++){
			for(int c=0;c<z.numColumns();c++){
				diff.set(r,c,z.get(r,c) - previousz.get(r,c));
			}
		}
		return numMappers*rho*rho*diff.norm2();
	}
}
