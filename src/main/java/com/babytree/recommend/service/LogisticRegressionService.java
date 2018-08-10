package com.babytree.recommend.service;

import org.apache.spark.SparkConf;
import org.apache.spark.SparkContext;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.JavaSparkContext;
import org.apache.spark.api.java.function.Function;
import org.apache.spark.mllib.classification.LogisticRegressionModel;
import org.apache.spark.mllib.classification.LogisticRegressionWithLBFGS;
import org.apache.spark.mllib.evaluation.MulticlassMetrics;
import org.apache.spark.mllib.regression.LabeledPoint;
import org.apache.spark.mllib.util.MLUtils;
import scala.Tuple2;

/**
 * Created by Sean on 2018/8/10 16:31
 */
public class LogisticRegressionService {

    public void calculate(String filePath){
        SparkConf conf = new SparkConf().setAppName("Collaborative Filtering Test").setMaster("local[2]");
        SparkContext sc = new SparkContext(conf);

        JavaRDD<LabeledPoint> data = MLUtils.loadLibSVMFile(sc, filePath).toJavaRDD();

// Split initial RDD into two... [60% training data, 40% testing data].
        JavaRDD<LabeledPoint>[] splits = data.randomSplit(new double[] {0.6, 0.4}, 11L);
        JavaRDD<LabeledPoint> training = splits[0].cache();
        JavaRDD<LabeledPoint> test = splits[1];

// Run training algorithm to build the model.
        final LogisticRegressionModel model = new LogisticRegressionWithLBFGS()
                .setNumClasses(10)
                .run(training.rdd());

// Compute raw scores on the test set.
        JavaRDD<Tuple2<Object, Object>> predictionAndLabels = test.map(
                new Function<LabeledPoint, Tuple2<Object, Object>>() {
                    public Tuple2<Object, Object> call(LabeledPoint p) {
                        Double prediction = model.predict(p.features());
                        return new Tuple2<Object, Object>(prediction, p.label());
                    }
                }
        );

// Get evaluation metrics.
        MulticlassMetrics metrics = new MulticlassMetrics(predictionAndLabels.rdd());
        double accuracy = metrics.accuracy();
        System.out.println("==================================");
        System.out.println("Accuracy = " + accuracy);

// Save and load model
//        model.save(sc, "target/tmp/javaLogisticRegressionWithLBFGSModel");
//        LogisticRegressionModel sameModel = LogisticRegressionModel.load(sc,"target/tmp/javaLogisticRegressionWithLBFGSModel");
    }

    public static void main(String[] args){
        LogisticRegressionService service = new LogisticRegressionService();
        service.calculate(System.getProperty("user.dir")+"/data/cf/u.data");
    }

}
