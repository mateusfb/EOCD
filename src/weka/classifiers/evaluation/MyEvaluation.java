package weka.classifiers.evaluation;

import weka.classifiers.CostMatrix;
import weka.core.Instances;

public class MyEvaluation extends Evaluation {

	private static final long serialVersionUID = -8125592372876066317L;

	public MyEvaluation(Instances data) throws Exception {
		super(data);
	}

	public MyEvaluation(Instances data, CostMatrix costMatrix) throws Exception {
		super(data, costMatrix);
	}
	
	@Override
	public double precision(int classIndex) {
		double correct = 0, total = 0;
	    for (int i = 0; i < m_NumClasses; i++) {
	      if (i == classIndex) {
	        correct += m_ConfusionMatrix[i][classIndex];
	      }
	      total += m_ConfusionMatrix[i][classIndex];
	    }
	    return total == 0 ? 0 : correct / total;
	}
}
