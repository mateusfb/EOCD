import java.io.File;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.PrintStream;
import java.util.ArrayList;
import java.util.Iterator;
import java.util.Random;

import weka.classifiers.Classifier;
import weka.classifiers.RandomizableIteratedSingleClassifierEnhancer;
import weka.classifiers.evaluation.Evaluation;
import weka.classifiers.evaluation.MyEvaluation;
import weka.classifiers.semi.MultiSemiAdaBoost;
import weka.classifiers.semi.SemiBoost;
import weka.classifiers.trees.J48;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.converters.ArffLoader;

public class MethodologySemi {
  private static final int SEED = 19;
  
  private static final int NUM_FOLDS = 10;
  
  private static final int NUM_ITER = 1;
  
  private static String OUTPUT_FILE_NAME = "output.csv";
  
  private static String FIELD_SEPARATOR = ";";
  
  private static Register register;
  
  public static void main(String[] args) throws Exception {
    File output = new File(OUTPUT_FILE_NAME);
    if (!output.exists())
      createOutput(); 
    for (Instances instances : loadInstances()) {
      System.out.println(instances.relationName());
      for (int iter = 0; iter < 1; iter++) {
        try {
          System.out.print(String.valueOf(iter) + ": ");
          instances.randomize(new Random(19L));
          instances.stratify(10);
          register = new Register();
          for (int i = 0; i < 10; i++) {
            System.out.print(String.valueOf(i) + " ");
            Instances train = new Instances(instances.trainCV(10, i));
            ArrayList<Double> labels = eraseLabels(train, 0.9D);
            Instances test = instances.testCV(10, i);
            MyEvaluation eval = new MyEvaluation(train);
            Classifier classifier = createClassifier(instances.numClasses());
            double time = System.currentTimeMillis();
            classifier.buildClassifier(train);
            eval.evaluateModel(classifier, test, new Object[0]);
            time = (System.currentTimeMillis() - time) / 1000.0D;
            restoreLabels(train, labels);
            writeResults(instances.relationName(), eval, classifier.getClass().getSimpleName(), i, train.numClasses(), time, iter);
          } 
          System.out.println();
          writeResultsAVG(instances.relationName(), createClassifier(instances.numClasses()).getClass().getSimpleName(), iter);
        } catch (Error e) {
          System.err.println("Error in classifier build for database (" + instances.relationName() + ")");
        } 
      } 
    } 
  }
  
  public static ArrayList<Instances> loadInstancesTest() throws Exception {
    ArrayList<Instances> instances = new ArrayList<>();
    instances.add(load("Arrhythmia.arff"));
    return instances;
  }
  
  public static ArrayList<Instances> loadInstances() throws Exception {
    ArrayList<Instances> instances = new ArrayList<>();
    instances.add(load("exec/Abalone.arff"));
    //instances.add(load("Adult.arff"));
    //instances.add(load("Arrhythmia.arff"));
    //instances.add(load("Automobile.arff"));
    //instances.add(load("Btsc.arff"));
    //instances.add(load("Car.arff"));
    //instances.add(load("Cnae.arff"));
    //instances.add(load("Dermatology.arff"));
    //instances.add(load("Ecoli.arff"));
    //instances.add(load("Flags.arff"));
    //instances.add(load("GermanCredit.arff"));
    //instances.add(load("Glass.arff"));
    //instances.add(load("Haberman.arff"));
    //instances.add(load("HillValley.arff"));
    //instances.add(load("Ilpd.arff"));
    //instances.add(load("ImageSegmentation_norm.arff"));
    //instances.add(load("KrVsKp.arff"));
    //instances.add(load("Leukemia.arff"));
    //instances.add(load("Madelon.arff"));
    //instances.add(load("MammographicMass.arff"));
    //instances.add(load("MultipleFeaturesKarhunen.arff"));
    //instances.add(load("Mushroom.arff"));
    //instances.add(load("Musk.arff"));
    //instances.add(load("Nursery.arff"));
    //instances.add(load("OzoneLevelDetection.arff"));
    //instances.add(load("PenDigits.arff"));
    //instances.add(load("PhishingWebsite.arff"));
    //instances.add(load("Pima.arff"));
    //instances.add(load("PlanningRelax.arff"));
    //instances.add(load("Seeds.arff"));
    //instances.add(load("Semeion.arff"));
    //instances.add(load("SolarFlare.arff"));
    //instances.add(load("SolarFlare1.arff"));
    //instances.add(load("Sonar.arff"));
    //instances.add(load("SpectfHeart.arff"));
    //instances.add(load("TicTacToeEndgame.arff"));
    //instances.add(load("Twonorm.arff"));
    //instances.add(load("Vehicle.arff"));
    //instances.add(load("Waveform.arff"));
    //instances.add(load("Wilt.arff"));
    instances.add(load("exec/Wine.arff"));
    //instances.add(load("Yeast.arff"));
    return instances;
  }
  
  private static Classifier createClassifier(int numClasses) {
    J48 baseClassifier = new J48();
    baseClassifier.setConfidenceFactor(0.05F);
    RandomizableIteratedSingleClassifierEnhancer classifier = (numClasses == 2) ? (RandomizableIteratedSingleClassifierEnhancer)new SemiBoost() : (RandomizableIteratedSingleClassifierEnhancer)new MultiSemiAdaBoost();
    classifier.setClassifier((Classifier)baseClassifier);
    return (Classifier)classifier;
  }
  
  private static void writeResultsAVG(String dataset, String method, int iter) throws Exception {
    File output = new File(OUTPUT_FILE_NAME);
    PrintStream out = new PrintStream(new FileOutputStream(output, true));
    out.print(dataset);
    out.print(FIELD_SEPARATOR);
    out.print(iter);
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
  
  private static void writeResults(String dataset, Evaluation eval, String method, int fold, int numClasses, double time, int iter) throws Exception {
    File output = new File(OUTPUT_FILE_NAME);
    PrintStream out = new PrintStream(new FileOutputStream(output, true));
    out.print(dataset);
    out.print(FIELD_SEPARATOR);
    out.print(iter);
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
    int i;
    for (i = 0; i < numClasses; i++) {
      out.print(FIELD_SEPARATOR);
      out.print(eval.precision(i));
    } 
    for (i = 0; i < numClasses; i++) {
      out.print(FIELD_SEPARATOR);
      out.print(eval.recall(i));
    } 
    for (i = 0; i < numClasses; i++) {
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
    out.print("Iteration");
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
    instances.randomize(new Random(0L));
    ArrayList<Double> labels = new ArrayList<>();
    int tenP = (int)Math.max(1.0D, instances.numInstances() * (1.0D - percent));
    for (int i = tenP; i < instances.numInstances(); i++) {
      Instance instance = instances.get(i);
      labels.add(Double.valueOf(instance.classValue()));
      instance.setClassMissing();
    } 
    return labels;
  }
  
  private static void restoreLabels(Instances instances, ArrayList<Double> labels) {
    int offset = instances.numInstances() - labels.size();
    for (Double label : labels) {
      Instance instance = instances.get(offset++);
      instance.setClassValue(label.doubleValue());
    } 
  }
  
  public static Instances load(String name) throws Exception {
    System.out.println("Loading " + name);
    ArffLoader loader = new ArffLoader();
    loader.setSource(new File(name));
    Instances instances = loader.getDataSet();
    instances.setClassIndex(instances.numAttributes() - 1);
    return instances;
  }
  
  private static class Register {
    private ArrayList<Double> accuracy = new ArrayList<>();
    
    private ArrayList<Double> precision = new ArrayList<>();
    
    private ArrayList<Double> recall = new ArrayList<>();
    
    private ArrayList<Double> fmeasure = new ArrayList<>();
    
    private ArrayList<Double> time = new ArrayList<>();
    
    public void register(double acc, double pre, double rec, double fme, double time) {
      this.accuracy.add(Double.valueOf(acc));
      this.precision.add(Double.valueOf(pre));
      this.recall.add(Double.valueOf(rec));
      this.fmeasure.add(Double.valueOf(fme));
      this.time.add(Double.valueOf(time));
    }
    
    private static double average(ArrayList<Double> values) {
      double m = 0.0D;
      int count = 0;
      for (Iterator<Double> iterator = values.iterator(); iterator.hasNext(); ) {
        double d = ((Double)iterator.next()).doubleValue();
        if (!Double.isNaN(d)) {
          m += d;
          count++;
        } 
      } 
      return m / count;
    }
    
    public double acc() {
      return average(this.accuracy);
    }
    
    public double pre() {
      return average(this.precision);
    }
    
    public double rec() {
      return average(this.recall);
    }
    
    public double fme() {
      return average(this.fmeasure);
    }
    
    public double time() {
      return average(this.time);
    }
  }
}