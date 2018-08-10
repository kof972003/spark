package com.babytree.recommend.service;

import org.apache.log4j.Level;
import org.apache.log4j.Logger;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.function.Function;
import org.apache.spark.ml.evaluation.RegressionEvaluator;
import org.apache.spark.ml.recommendation.ALSModel;
import org.apache.spark.sql.Dataset;
import org.apache.spark.sql.Row;
import org.apache.spark.sql.SparkSession;

import java.io.Serializable;

/**
 * Created by Sean on 2018/8/10 15:35
 */
public class CfService implements Serializable {

    //private JavaSparkContext sparkContext;
    private SparkSession sparkSession;

    public void init(){
        //if(sparkContext == null){
        //    SparkConf sparkConf = new SparkConf().setAppName("AdExposureMonitor");
        //    sparkContext = new JavaSparkContext(sparkConf);
        //}
        sparkSession = SparkSession.builder().appName("test").master("local[2]").config("spark.driver.memory","2147480000").getOrCreate();
    }

    public void calculate(String filePath){
        init();
        JavaRDD<com.babytree.recommend.domain.Rating> ratingsRDD = sparkSession.read().textFile(filePath).javaRDD()
                .map(new Function<String, com.babytree.recommend.domain.Rating>() {
                    public com.babytree.recommend.domain.Rating call(String str) {
                        return com.babytree.recommend.domain.Rating.parseRating(str);
                    }
                });
        Dataset<Row> ratings = sparkSession.createDataFrame(ratingsRDD, com.babytree.recommend.domain.Rating.class);
        Dataset<Row>[] splits = ratings.randomSplit(new double[]{0.0008, 0.0002,0.999});
        Dataset<Row> training = splits[0];
        Dataset<Row> test = splits[1];
        //Dataset<Row> training = ratings.limit(1000);

        // Build the recommendation model using ALS on the training data
        org.apache.spark.ml.recommendation.ALS als = new org.apache.spark.ml.recommendation.ALS()
                .setMaxIter(5)
                .setRegParam(0.01)
                .setImplicitPrefs(true)
                .setUserCol("userId")
                .setItemCol("movieId")
                .setRatingCol("rating");
        ALSModel model = als.fit(training);
        // Evaluate the model by computing the RMSE on the test data
        Dataset<Row> predictions = model.transform(test);

        RegressionEvaluator evaluator = new RegressionEvaluator()
                .setMetricName("rmse")
                .setLabelCol("rating")
                .setPredictionCol("prediction");
        Double rmse = evaluator.evaluate(predictions);
        System.out.println("Root-mean-square error = " + rmse);
    }

    public static void main(String[] args){
        Logger.getLogger("org").setLevel(Level.ERROR);
        CollaborativeFilteringService service = new CollaborativeFilteringService();
        //service.calculate(System.getProperty("user.dir")+"/data/cf/ratings.dat");
        service.calculate(System.getProperty("user.dir")+"/data/cf/u.data");
    }

}
