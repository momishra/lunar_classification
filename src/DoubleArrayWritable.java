import no.uib.cipr.matrix.DenseMatrix;

import org.apache.hadoop.io.DoubleWritable;
import org.apache.hadoop.io.TwoDArrayWritable;

public class DoubleArrayWritable extends TwoDArrayWritable {
    public DoubleArrayWritable() {
        super(DoubleWritable.class);
    }

	public static DoubleArrayWritable getWritable(DenseMatrix val) {
		DoubleWritable[][] data = new DoubleWritable[val.numRows()][val.numColumns()];
		for (int k = 0; k < data.length; k++) {
			for (int j = 0; j < data[k].length; j++) {
				data[k][j] = new DoubleWritable(val.get(k, j));
			}
		}
		DoubleArrayWritable wrt = new DoubleArrayWritable();
		wrt.set(data); 
		return wrt;
	}
}