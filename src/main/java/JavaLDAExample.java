// $example on$
import breeze.linalg.DenseVector;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.ml.clustering.LDA;
import org.apache.spark.ml.clustering.LDAModel;
import org.apache.spark.ml.linalg.BLAS;
import org.apache.spark.ml.linalg.Vector;
import org.apache.spark.ml.stat.Summarizer;
import org.apache.spark.sql.*;
import org.apache.spark.sql.api.java.UDF2;
import org.apache.spark.sql.types.DataTypes;

import static org.apache.spark.sql.functions.avg;
import static org.apache.spark.sql.functions.col;
// $example off$

/**
 * An example demonstrating LDA.
 * Run with
 * <pre>
 * bin/run-example ml.JavaLDAExample
 * </pre>
 */
public class JavaLDAExample {

    public static void main(String[] args) {
        // Creates a SparkSession
        SparkSession spark = SparkSession
                .builder()
                .appName("JavaLDAExample")
                .getOrCreate();

        // $example on$
        // Loads data.
        Dataset<Row> dataset = spark.read().format("libsvm")
                .load("/usr/project/SimpleProject/data/sample_lda_libsvm_data.txt");
        dataset.printSchema();

        // Trains a LDA model.
        LDA lda = new LDA().setK(10).setMaxIter(10);
        LDAModel model = lda.fit(dataset);

        double ll = model.logLikelihood(dataset);
        double lp = model.logPerplexity(dataset);
        System.out.println("The lower bound on the log likelihood of the entire corpus: " + ll);
        System.out.println("The upper bound on perplexity: " + lp);

        // Describe topics.
        Dataset<Row> topics = model.describeTopics(3);
        System.out.println("The topics described by their top-weighted terms:");
        topics.show(false);

        // Shows the result.
        Dataset<Row> transformed = model.transform(dataset);
        transformed.printSchema();
        transformed = transformed.drop("features");
        transformed.show(false);
        // $example off$
        transformed.printSchema();

        Dataset<Row> input = spark.read().format("libsvm")
                .load("/usr/project/SimpleProject/data/input.txt");
        input = transformed.join(input, "label").drop("features");
        input.show(false);
        input.select("topicDistribution").foreach(v->System.out.println(v.fieldIndex("topicDistribution")));
        input.select("topicDistribution").foreach(v->System.out.println(v.getAs("topicDistribution").getClass()));
        input.agg(avg("label")).show();

        input.groupBy().agg(Summarizer.mean(col("topicDistribution"))).show();

        Dataset<Row> table= transformed.except(input);
        table.show();
        table = table.join(input.groupBy().agg(Summarizer.mean(col("topicDistribution"))));
        table.show();

        spark.udf().register("cos_func",
                (Vector v1, Vector v2)-> BLAS.dot(v1, v2)/(Math.sqrt(BLAS.dot(v1,v1))*Math.sqrt(BLAS.dot(v2,v2))),
                DataTypes.DoubleType);
        table.withColumn("cosine",
                functions.callUDF("cos_func", col("topicDistribution"), col("mean(topicDistribution)")))
                .show();
//        table.map(row -> {
//            Vector v1 = (DenseVector) row.get(row.fieldIndex("topicDistribution"));
//            Vector v2 = (DenseVector) row.get(row.fieldIndex("mean(topicDistribution)"));
//            double v = v1.
//
//            return v;
//        }, Encoders.DOUBLE()).show();



//        try {
//            model.write().overwrite().save("/usr/project/SimpleProject/model/LDAmodel2");
//            transformed.write().parquet("/usr/project/SimpleProject/model/docRepresentation.parquet");
//        } catch (IOException e) {
//            e.printStackTrace();
//        }

        spark.stop();
    }
}
