import java.io.File;
import java.io.IOException;
import java.util.Random;

import org.datavec.api.io.labels.ParentPathLabelGenerator;
import org.datavec.api.records.reader.RecordReader;
import org.datavec.api.split.FileSplit;
import org.datavec.image.loader.NativeImageLoader;
import org.datavec.image.recordreader.ImageRecordReader;
import org.deeplearning4j.datasets.datavec.RecordReaderDataSetIterator;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.util.ModelSerializer;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.dataset.api.preprocessor.DataNormalization;
import org.nd4j.linalg.dataset.api.preprocessor.ImagePreProcessingScaler;

public class Color_Prediction {

	public static void main(String[] args) throws IOException, InterruptedException {
		String[] label = {"Blue" , "Green","Red"} ;
		 MultiLayerNetwork model = ModelSerializer.restoreMultiLayerNetwork
				 (new File("ColorModel.zip"));
			String path ="colors_cnn";
			File filePred = new File(path+"/prediction");
			FileSplit fileSplitPred = new FileSplit(filePred , NativeImageLoader.ALLOWED_FORMATS,new Random(1234));
			RecordReader recordReaderPred = new ImageRecordReader(100,100,3,new ParentPathLabelGenerator());
			recordReaderPred.initialize(fileSplitPred);
			
			DataSetIterator dataSetIteratorPred = new RecordReaderDataSetIterator(recordReaderPred, 1,1,3);
			DataNormalization scaler = new ImagePreProcessingScaler(0,1);
			dataSetIteratorPred.setPreProcessor(scaler);
			
			while(dataSetIteratorPred.hasNext()) {
				DataSet dataset = dataSetIteratorPred.next();
				INDArray features = dataset.getFeatures();
				INDArray predicted = model.output(features);
				int[] classes = predicted.argMax(1).toIntVector();
				System.out.println(predicted);
				for(int i=0 ; i<classes.length;i++) {
					System.out.println("Couleur : "+ label[classes[i]]);
				}
			}

	}

}
