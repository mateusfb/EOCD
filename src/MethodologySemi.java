import java.io.File;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.PrintStream;
import java.util.ArrayList;
import java.util.Random;

import weka.classifiers.Classifier;
import weka.classifiers.RandomizableIteratedSingleClassifierEnhancer;
import weka.classifiers.evaluation.Evaluation;
import weka.classifiers.semi.MultiSemiAdaBoost;
import weka.classifiers.semi.SemiBoost;
import weka.classifiers.trees.J48;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.converters.ArffLoader;

public class MethodologySemi {

	private static final int SEED = 19;
	private static final int NUM_FOLDS = 10;
	private static String OUTPUT_FILE_NAME = "output.csv";
	private static String FIELD_SEPARATOR = ";";

	private static Register register;
	
	public static void main(String[] args) throws Exception {
		File output = new File(OUTPUT_FILE_NAME);
		if (!output.exists()) {
			createOutput();
		}
		
		for (Instances instances : loadInstances()) {
			instances.randomize(new Random(SEED));
			instances.stratify(10);
			register = new Register();
			for (int i = 0; i < NUM_FOLDS; i++) {
				Instances train = new Instances(instances.trainCV(NUM_FOLDS, i));
				ArrayList<Double> labels = eraseLabels(train, 0.9);
				Instances test = instances.testCV(NUM_FOLDS, i);
				Evaluation eval = new Evaluation(train);
				Classifier classifier = createClassifier(instances.numClasses());
				double time = System.currentTimeMillis();
				classifier.buildClassifier(train);
				eval.evaluateModel(classifier, test);
				time = (System.currentTimeMillis() - time) / 1000.0;
				restoreLabels(train, labels);
				writeResults(instances.relationName(), eval, classifier.getClass().getSimpleName(), i, train.numClasses(), time);
			}
			writeResultsAVG(instances.relationName(), createClassifier(instances.numClasses()).getClass().getSimpleName());
		}
	}
	
	public static ArrayList<Instances> loadInstances() throws Exception {
		ArrayList<Instances> instances = new ArrayList<>();
		instances.add(load("Iris.arff"));
		instances.add(load("Balance.arff"));
		instances.add(load("Dermatology.arff"));
		instances.add(load("Vehicle23.arff"));
		instances.add(load("Glass Identification.arff"));
		instances.add(load("Labor.arff"));
		instances.add(load("Lung Cancer.arff"));
		return instances;
	}
	
	private static Classifier createClassifier(int numClasses) {
		J48 baseClassifier = new J48();
		baseClassifier.setConfidenceFactor(0.05f);
		RandomizableIteratedSingleClassifierEnhancer classifier = numClasses == 2 ? new SemiBoost() : new MultiSemiAdaBoost();
		classifier.setClassifier(baseClassifier);
		return classifier;
	}
	
	private static void writeResultsAVG(String dataset, String method) throws Exception {
		File output = new File(OUTPUT_FILE_NAME);
		PrintStream out = new PrintStream(new FileOutputStream(output, true));
		out.print(dataset);
		out.print(FIELD_SEPARATOR);
		out.print('-');
		out.print(FIELD_SEPARATOR);
		out.print(method);
		out.print(FIELD_SEPARATOR);
		out.print(register.acc());
		out.print(FIELD_SEPARATOR);
		out.print(register.pre());
		out.print(FIELD_SEPARATOR);
		out.print(register.rec());
		out.print(FIELD_SEPARATOR);
		out.print(register.fme());
		out.print(FIELD_SEPARATOR);
		out.print(register.time());
		out.print("\n");
		out.close();
	}

	private static void writeResults(String dataset, Evaluation eval, String method, int fold, int numClasses, double time) throws Exception {
		File output = new File(OUTPUT_FILE_NAME);
		PrintStream out = new PrintStream(new FileOutputStream(output, true));
		out.print(dataset);
		out.print(FIELD_SEPARATOR);
		out.print(fold);
		out.print(FIELD_SEPARATOR);
		out.print(method);
		out.print(FIELD_SEPARATOR);
		out.print(eval.pctCorrect());
		out.print(FIELD_SEPARATOR);
		out.print(eval.weightedPrecision());
		out.print(FIELD_SEPARATOR);
		out.print(eval.weightedRecall());
		out.print(FIELD_SEPARATOR);
		out.print(eval.weightedFMeasure());
		out.print(FIELD_SEPARATOR);
		out.print(time);
		register.register(eval.pctCorrect(), eval.weightedPrecision(), eval.weightedRecall(), eval.weightedFMeasure(), time);
		for (int i = 0; i < numClasses; i++) {
			out.print(FIELD_SEPARATOR);
			out.print(eval.precision(i));
		}
		for (int i = 0; i < numClasses; i++) {
			out.print(FIELD_SEPARATOR);
			out.print(eval.recall(i));
		}
		for (int i = 0; i < numClasses; i++) {
			out.print(FIELD_SEPARATOR);
			out.print(eval.fMeasure(i));
		}
		out.print("\n");
		out.close();
	}

	private static void createOutput() throws IOException {
		PrintStream out = new PrintStream(OUTPUT_FILE_NAME);
		out.print("Dataset");
		out.print(FIELD_SEPARATOR);
		out.print("Fold");
		out.print(FIELD_SEPARATOR);
		out.print("Method");
		out.print(FIELD_SEPARATOR);
		out.print("Accuracy");
		out.print(FIELD_SEPARATOR);
		out.print("Precision");
		out.print(FIELD_SEPARATOR);
		out.print("Recall");
		out.print(FIELD_SEPARATOR);
		out.print("F-Measure");
		out.print(FIELD_SEPARATOR);
		out.print("Seconds");
		out.print(FIELD_SEPARATOR);
		out.print("Precision by Class");
		out.print(FIELD_SEPARATOR);
		out.print("Recall by Class");
		out.print(FIELD_SEPARATOR);
		out.print("F-Measure by Class");
		out.print("\n");
		out.close();
	}

	public static ArrayList<Double> eraseLabels(Instances instances, double percent) {
		assert instances.numInstances() > 1;
		instances.randomize(new Random(0));
		ArrayList<Double> labels = new ArrayList<>();
		int tenP = (int) Math.max(1, instances.numInstances() * (1.0-percent));
		for (int i = tenP; i < instances.numInstances(); i++) {
			Instance instance = instances.get(i);
			labels.add(instance.classValue());
			instance.setClassMissing();
		}
		return labels;
	}

	private static void restoreLabels(Instances instances, ArrayList<Double> labels) {
		int offset = instances.numInstances() - labels.size();
		for (Double label : labels) {
			Instance instance = instances.get(offset++);
			instance.setClassValue(label);
		}
	}

	public static Instances load(String name) throws Exception {
		ArffLoader loader = new ArffLoader();
		loader.setSource(new File(name));
		Instances instances = loader.getDataSet();
		instances.setClassIndex(instances.numAttributes() - 1);
		return instances;
	}
	
	private static class Register {
		private ArrayList<Double> accuracy;
		private ArrayList<Double> precision;
		private ArrayList<Double> recall;
		private ArrayList<Double> fmeasure;
		private ArrayList<Double> time;
		
		public Register() {
			accuracy = new ArrayList<>();
			precision = new ArrayList<>();
			recall = new ArrayList<>();
			fmeasure = new ArrayList<>();
			time = new ArrayList<>();
		}
		
		public void register(double acc, double pre, double rec, double fme, double time) {
			accuracy.add(acc);
			precision.add(pre);
			recall.add(rec);
			fmeasure.add(fme);
			this.time.add(time);
		}
		
		private static double average(ArrayList<Double> values) {
			double m = 0;
			int count = 0;
			for(double d : values) {
				if(!Double.isNaN(d)) {
					m += d;
					count += 1;
				}
			}
			return m / count;
		}
		
		public double acc() {
			return average(accuracy);
		}
		
		public double pre() {
			return average(precision);
		}
		
		public double rec() {
			return average(recall);
		}
		
		public double fme() {
			return average(fmeasure);
		}
		
		public double time() {
			return average(time);
		}
	}
}
