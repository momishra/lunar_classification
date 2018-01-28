import java.io.IOException;
import java.lang.reflect.InvocationTargetException;
import java.net.URI;
import java.net.URISyntaxException;
import no.uib.cipr.matrix.DenseMatrix;
import no.uib.cipr.matrix.Matrices;
import org.apache.commons.logging.Log;
import org.apache.commons.logging.LogFactory;
import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.conf.Configured;
import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.DoubleWritable;
import org.apache.hadoop.io.LongWritable;
import org.apache.hadoop.io.MapWritable;
import org.apache.hadoop.mapreduce.Counters;
import org.apache.hadoop.mapreduce.filecache.DistributedCache;
import org.apache.hadoop.mapreduce.lib.output.SequenceFileOutputFormat;
import org.apache.hadoop.mapreduce.Job;
import org.apache.hadoop.mapreduce.lib.input.FileInputFormat;
import org.apache.hadoop.mapreduce.lib.output.FileOutputFormat;
import org.apache.hadoop.util.Tool;
import org.apache.hadoop.util.ToolRunner;
import org.apache.zookeeper.KeeperException;
import com.google.gson.Gson;
/* 
 * @author Mohini Mishra 
 */

public class tempUtility extends Configured implements Tool {
	private int NumberofHiddenNeurons = 300; 
	private int NumberofOutputNeurons = 3; 
	public static DenseMatrix InputWeight;
	public static DenseMatrix BiasofHiddenNeurons;
	public static int it=0;
	private static final Log LOG = LogFactory.getLog(tempUtility.class.getName());
	 
	 public static void main(String[] args) throws Exception {
	      int exitCode = ToolRunner.run(new Configuration(), new tempUtility(), args);
	      System.exit(exitCode);
	  }
	 
  public int run(String args[])throws IOException, ClassNotFoundException, InterruptedException, URISyntaxException, KeeperException, InstantiationException, IllegalAccessException, IllegalArgumentException, InvocationTargetException, NoSuchMethodException, SecurityException{
	    ZKClientManagerImpl zk = new ZKClientManagerImpl();		
		DenseMatrix tempInput = (DenseMatrix) Matrices.random(NumberofHiddenNeurons,72);
		InputWeight = new DenseMatrix(NumberofHiddenNeurons,72);
		for(int r=0;r<tempInput.numRows();r++){
			for(int c=0;c<tempInput.numColumns();c++){
				InputWeight.set(r,c,tempInput.get(r,c)*2 -1);
			}
		}
		BiasofHiddenNeurons = (DenseMatrix) Matrices.random(NumberofHiddenNeurons, 1);
		
		Gson gson = new Gson();
		if(ZKClientManagerImpl.zkeeper.exists("/inputWeight", false) != null ||
				ZKClientManagerImpl.zkeeper.exists("/bias", false) != null){
			zk.delete("/inputWeight");
			zk.delete("/bias");
		}
		zk.create("/inputWeight",gson.toJson(InputWeight).getBytes());
		zk.create("/bias",gson.toJson(BiasofHiddenNeurons).getBytes());
		
		/*job creation begins here*/
		Configuration conf = this.getConf();
		conf.setInt("NumberofHiddenNeurons", NumberofHiddenNeurons);
    	conf.setInt("NumberofOutputNeurons", NumberofOutputNeurons);
    	conf.set("mapreduce.job.reduce.slowstart.completedmaps", "0" );//range 0 to 1 only.
    	
    	Path inPath = new Path(args[0]);
    	Path outPath = new Path(args[1]);
    	
		FileSystem hdfs = FileSystem.get(conf);
		Job job1 = new Job(conf,"ParameterCalc");
		job1.setMapperClass(firstMapper.class); 
		job1.setMapOutputKeyClass(LongWritable.class);
		job1.setMapOutputValueClass(MapWritable.class);
		job1.setOutputKeyClass(LongWritable.class);
		job1.setOutputValueClass(MapWritable.class);
		job1.setInputFormatClass(MultiLineInputFormat.class);
		job1.setOutputFormatClass(SequenceFileOutputFormat.class);
		job1.setJar("/home/hduser/Desktop/AsyncHadoopELM.jar");

		// delete existing directory
		if (hdfs.exists(outPath)) {
		    hdfs.delete(outPath, true);
		}
		FileInputFormat.addInputPath(job1, inPath);
	    FileOutputFormat.setOutputPath(job1, outPath);
	    
	    Job job2 =null;
	    int i = job1.waitForCompletion(true) ? 0 :1;
	    
	    LOG.info("Checking job for completion thing :"+ i);
	    
	    if (job1.waitForCompletion(true)){
	    	LOG.info("Am i even getting inside this? --if control");
 	    	conf = new Configuration();
	    	inPath = outPath;
	    	DistributedCache.addCacheFile(new URI(inPath + "/part-r-00000"), conf);
			int iterationNumber = 0;
			boolean isFinalIteration = false;
			
			while (!isFinalIteration) {
				conf.setInt("iterCount",iterationNumber);
				outPath = new Path("trainOut_"+iterationNumber); //have to change this
				if (hdfs.exists(outPath)) {
					hdfs.delete(outPath, true);
				}
				job2 = new Job(conf, "OutputWeightCalc");
				job2.setJar("/home/hduser/Desktop/AsyncHadoopELM.jar"); //change this path
				job2.setMapperClass(secMapper.class);
				job2.setMapOutputKeyClass(LongWritable.class);
				job2.setMapOutputValueClass(MapWritable.class);
				job2.setReducerClass(finalReducer.class);
				job2.setOutputKeyClass(LongWritable.class);
				job2.setOutputValueClass(MapWritable.class);
				job2.setInputFormatClass(CustomSequenceFileSplit.class);
				job2.setOutputFormatClass(SequenceFileOutputFormat.class);
				job2.setNumReduceTasks(1);
				FileInputFormat.setInputPaths(job2, inPath);
				SequenceFileOutputFormat.setOutputPath(job2, outPath);
				job2.waitForCompletion(false);
				Counters count = job2.getCounters();
				if(count.findCounter("iterationCounter", "iteration").getValue() != 0){
					isFinalIteration = true;
				}
				inPath = outPath;
				iterationNumber++;
			}
	    }
	    /* Training phase ends here and testing phase begins */
	    
		if(job2.waitForCompletion(true)){
			conf = new Configuration();
			conf.setInt("NumberofHiddenNeurons", NumberofHiddenNeurons);
	    	conf.setInt("NumberofOutputNeurons", NumberofOutputNeurons);
			DistributedCache.addCacheFile(new URI(outPath+"/part-r-00000"), conf);

			inPath = new Path(args[2]);
			outPath = new Path(args[3]);
			if (hdfs.exists(outPath)) {
				hdfs.delete(outPath, true);
			}
			Job job3 = new Job(conf, "Testing");
			job3.setJar("/home/hduser/Desktop/AsyncHadoopELM.jar"); //change this here also
			job3.setMapperClass(testMapper.class);
			job3.setMapOutputKeyClass(LongWritable.class);
			job3.setMapOutputValueClass(DoubleWritable.class);
			job3.setOutputKeyClass(LongWritable.class);
			job3.setOutputValueClass(DoubleWritable.class);
			job3.setInputFormatClass(MultiLineInputFormat.class);
			FileInputFormat.addInputPath(job3, inPath);
			FileOutputFormat.setOutputPath(job3, outPath);
			job3.waitForCompletion(true);	
		}
	
		return 0;
	}

  	//this is for making hadoop asynchronous
	/*private void setParameters(Configuration conf) {
		conf.set("mapreduce.job.reduce.slowstart.completedmaps", "0" );
		conf.setBoolean("mapreduce.sort.avoidance", true);
		conf.setInt("NumberofHiddenNeurons", NumberofHiddenNeurons);
    	conf.setInt("NumberofOutputNeurons", NumberofOutputNeurons);
    	conf.setDouble("rho",0.9);
    	conf.setClass("mapreduce.job.map.output.collector.class",CustomMapOutputBuffer.class, MapOutputCollector.class);
    	conf.setClass("mapreduce.job.reduce.shuffle.consumer.plugin.class", CustomShuffle.class, ShuffleConsumerPlugin.class);
    	conf.setBoolean("noMapOutputFile",true);
	
	}*/
}
