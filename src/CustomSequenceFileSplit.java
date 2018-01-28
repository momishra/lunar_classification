

import java.io.IOException;
import java.util.ArrayList;
import java.util.List;
import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.FileStatus;
import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.LongWritable;
import org.apache.hadoop.io.MapWritable;
import org.apache.hadoop.io.SequenceFile;
import org.apache.hadoop.mapreduce.InputSplit;
import org.apache.hadoop.mapreduce.JobContext;
import org.apache.hadoop.mapreduce.RecordReader;
import org.apache.hadoop.mapreduce.TaskAttemptContext;
import org.apache.hadoop.mapreduce.lib.input.FileInputFormat;
import org.apache.hadoop.mapreduce.lib.input.FileSplit;

public class CustomSequenceFileSplit extends FileInputFormat<LongWritable, MapWritable> {
	
	@Override
	public RecordReader<LongWritable, MapWritable> createRecordReader(InputSplit genericSplit, TaskAttemptContext context)	throws IOException {
		context.setStatus(genericSplit.toString());
		return new SequenceRecordReader();
	}
	
	public static class SequenceRecordReader extends RecordReader<LongWritable, MapWritable>{

		private SequenceFile.Reader in;
	    private long start;
	    private long end;
	    private boolean done = false;
	    private LongWritable key;
	    private MapWritable value;
	    long pos;
	    
		@Override
		public void close() throws IOException {
			 if (in != null) {
	                in.close();
	            }
		}

		@Override
		public LongWritable getCurrentKey() throws IOException, InterruptedException {
			return this.key;
		}

		@Override
		public MapWritable getCurrentValue() throws IOException,InterruptedException {
			return this.value;
		}

		@Override
		public float getProgress() throws IOException, InterruptedException {
		      if (end == start) {
		          return 0.0f;
		        } else {
		          return Math.min(1.0f, (float)((in.getPosition() - start) /(double)(end - start)));
		        }
		      }

		@Override
		public void initialize(InputSplit genericSplit, TaskAttemptContext context) throws IOException, InterruptedException {FileSplit split = (FileSplit) genericSplit; 
		FileSplit splits = (FileSplit) genericSplit; 
		Path path = splits.getPath();
        Configuration conf = context.getConfiguration();
        FileSystem fs = path.getFileSystem(conf);
        start = splits.getStart();
        this.in = new SequenceFile.Reader(fs, path, conf);
        this.end = split.getStart() + split.getLength();
        if (split.getStart() > in.getPosition()){
        	 in.sync(split.getStart());  // sync to start
        }              
        this.start = in.getPosition();
        done = start >= end;
        this.pos = start;
        }

		public boolean nextKeyValue() throws IOException, InterruptedException {
			if (done) return false;
			if(key == null){
				key = new LongWritable();
			}
			if(value == null){
				value = new MapWritable();
			}
			
			pos = in.getPosition();
			if(pos< end){
				if (!in.next(key, value)) {
					 return false;	
			}
			}else{
				return false;
			}
			return true;
		}
	}
	
	public List<InputSplit> getSplits(JobContext job) throws IOException {  
		List<InputSplit> splits = new ArrayList<InputSplit>();
		for (FileStatus status : listStatus(job)) {
			try {
				splits.addAll(getSplitsForFile(status, job.getConfiguration()));
			} catch (IllegalAccessException e) {
				System.out.println(e.getMessage());
			}
		}
		return splits;
	}
	
	public static List<FileSplit> getSplitsForFile(FileStatus status,Configuration conf) throws IOException, IllegalAccessException {
		/*implements the logic of how input will be distributed between the map processes.*/
		List<FileSplit> splits = new ArrayList<FileSplit>();
		Path fileName = status.getPath();
		if (status.isDir()) {
			throw new IOException("Not a file: " + fileName);
		}
		FileSystem fs = fileName.getFileSystem(conf);
		SequenceFile.Reader reader = new SequenceFile.Reader(fs, fileName, conf);	
			
			try {
				LongWritable keys = (LongWritable) reader.getKeyClass().newInstance();
				long begin = reader.getPosition();
				long length = 0;
				while(reader.next(keys)){
					long end = reader.getPosition();
					length = end - begin;
					splits.add(new FileSplit(fileName, begin, length -1, new String[] {}));
					begin = 0;
					begin+= end;
					length = 0;
				}
			} catch (InstantiationException e) {
				System.out.println(e.getMessage());
			}
			finally{
				reader.close();
			}
		return splits;
	}
}
