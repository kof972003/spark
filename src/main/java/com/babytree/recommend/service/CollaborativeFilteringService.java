package com.babytree.recommend.service;

import org.apache.spark.SparkConf;
import org.apache.spark.api.java.JavaDoubleRDD;
import org.apache.spark.api.java.JavaPairRDD;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.JavaSparkContext;
import org.apache.spark.api.java.function.Function;
import org.apache.spark.mllib.recommendation.MatrixFactorizationModel;
import scala.Tuple2;

import java.io.Serializable;

/**
 * Created by Sean on 2018/8/9 17:15
 */
public class CollaborativeFilteringService implements Serializable {

    public void calculate(String filePath){
        SparkConf conf = new SparkConf().setAppName("Collaborative Filtering Test").setMaster("local[2]");
        JavaSparkContext jsc = new JavaSparkContext(conf);
        JavaRDD<String> data = jsc.textFile(filePath);
        JavaRDD<org.apache.spark.mllib.recommendation.Rating> ratings = data.map(
                new Function<String, org.apache.spark.mllib.recommendation.Rating>() {
                    public org.apache.spark.mllib.recommendation.Rating call(String s) {
                        String[] sarray = s.split("\\t");
                        return new org.apache.spark.mllib.recommendation.Rating(Integer.parseInt(sarray[0]), Integer.parseInt(sarray[1]),
                                Float.parseFloat(sarray[2]));
                    }
                }
        );

// Build the recommendation model using ALS
        int rank = 10;
        int numIterations = 10;
        MatrixFactorizationModel model = org.apache.spark.mllib.recommendation.ALS.train(JavaRDD.toRDD(ratings), rank, numIterations, 0.01);

// Evaluate the model on rating data
        JavaRDD<Tuple2<Object, Object>> userProducts = ratings.map(
                new Function<org.apache.spark.mllib.recommendation.Rating, Tuple2<Object, Object>>() {
                    public Tuple2<Object, Object> call(org.apache.spark.mllib.recommendation.Rating r) {
                        return new Tuple2<Object, Object>(r.user(), r.product());
                    }
                }
        );
        JavaPairRDD<Tuple2<Integer, Integer>, Double> predictions = JavaPairRDD.fromJavaRDD(
                model.predict(JavaRDD.toRDD(userProducts)).toJavaRDD().map(
                        new Function<org.apache.spark.mllib.recommendation.Rating, Tuple2<Tuple2<Integer, Integer>, Double>>() {
                            public Tuple2<Tuple2<Integer, Integer>, Double> call(org.apache.spark.mllib.recommendation.Rating r){
                                return new Tuple2<>(new Tuple2<>(r.user(), r.product()), r.rating());
                            }
                        }
                ));
        JavaRDD<Tuple2<Double, Double>> ratesAndPreds =
                JavaPairRDD.fromJavaRDD(ratings.map(
                        new Function<org.apache.spark.mllib.recommendation.Rating, Tuple2<Tuple2<Integer, Integer>, Double>>() {
                            public Tuple2<Tuple2<Integer, Integer>, Double> call(org.apache.spark.mllib.recommendation.Rating r){
                                return new Tuple2<>(new Tuple2<>(r.user(), r.product()), r.rating());
                            }
                        }
                )).join(predictions).values();
        double MSE = JavaDoubleRDD.fromRDD(ratesAndPreds.map(
                new Function<Tuple2<Double, Double>, Object>() {
                    public Object call(Tuple2<Double, Double> pair) {
                        Double err = pair._1() - pair._2();
                        return err * err;
                    }
                }
        ).rdd()).mean();
        System.out.println("Mean Squared Error = " + MSE);
        System.out.println("==================================");
        System.out.println("start recommend users : ");
        org.apache.spark.mllib.recommendation.Rating[] userResults = model.recommendUsers(387,10);
        for(org.apache.spark.mllib.recommendation.Rating rating : userResults){
            System.out.println(rating.user()+","+rating.product()+","+rating.rating());
        }
        System.out.println("end recommend users : ");
        System.out.println("==================================");
        System.out.println("start recommend products : ");
        org.apache.spark.mllib.recommendation.Rating[] productResults = model.recommendProducts(387,10);
        for(org.apache.spark.mllib.recommendation.Rating rating : productResults){
            System.out.println(rating.user()+","+rating.product()+","+rating.rating());
        }
        System.out.println("end recommend products : ");
// Save and load model
//        model.save(jsc.sc(), "/user/xlq/hadoop/myCollaborativeFilter");
//        System.out.println("start load model");
//        System.out.println("==================================");
//        MatrixFactorizationModel sameModel = MatrixFactorizationModel.load(jsc.sc(),"/user/xlq/hadoop/myCollaborativeFilter");
//        System.out.println("end load model");
//        System.out.println("==================================");
//        System.out.println("start recommend users with saved model : ");
//        org.apache.spark.mllib.recommendation.Rating[] userResults2 = sameModel.recommendUsers(603,10);
//        for(org.apache.spark.mllib.recommendation.Rating rating : userResults2){
//            System.out.println(rating.user()+","+rating.product()+","+rating.rating());
//        }
//        System.out.println("end recommend users with saved model : ");
    }

    public static void main(String[] args){
        CfService service = new CfService();
        service.calculate(System.getProperty("user.dir")+"/data/cf/u.data");
    }

}
