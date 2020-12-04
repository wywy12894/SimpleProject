import org.apache.spark.ml.clustering.DistributedLDAModel;
import org.apache.spark.ml.clustering.LDA;
import org.apache.spark.ml.clustering.LDAModel;
import org.apache.spark.ml.clustering.LocalLDAModel;
import org.apache.spark.ml.linalg.BLAS;
import org.apache.spark.ml.linalg.Vector;
import org.apache.spark.ml.stat.Summarizer;
import org.apache.spark.sql.Dataset;
import org.apache.spark.sql.Row;
import org.apache.spark.sql.SparkSession;
import org.apache.spark.sql.functions;
import org.apache.spark.sql.types.DataTypes;

import java.io.IOException;

import static org.apache.spark.sql.functions.col;
import static org.apache.spark.sql.functions.max;

public class Main {
    public static void main(String[] args) {
        // Creates a SparkSession
        SparkSession spark = SparkSession
                .builder()
                .appName("Main")
                .getOrCreate();

        // Load document vector
        Dataset<Row> documents = spark.read().parquet("/usr/project/SimpleProject/model/docRepresentation.parquet");
        // Load input
        Dataset<Row> input = spark.read().format("libsvm").load("/usr/project/SimpleProject/data/input.txt");
        // input vector
        input = documents.join(input,"label").drop("features");
        // other documents
        documents = documents.except(input).join(input.groupBy().agg(Summarizer.mean(col("topicDistribution"))));
        // cosine distance
        spark.udf().register("cos_func",
                (Vector v1, Vector v2)-> BLAS.dot(v1, v2)/(Math.sqrt(BLAS.dot(v1,v1))*Math.sqrt(BLAS.dot(v2,v2))),
                DataTypes.DoubleType);
        documents = documents.withColumn("cosine",
                functions.callUDF("cos_func", col("topicDistribution"), col("mean(topicDistribution)")));
        // sort by cosine distance
        documents.sort(col("cosine")).limit(5).show();



        spark.stop();
    }
}
