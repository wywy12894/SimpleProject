import org.apache.spark.ml.clustering.DistributedLDAModel;
import org.apache.spark.ml.clustering.LDA;
import org.apache.spark.ml.clustering.LDAModel;
import org.apache.spark.sql.Dataset;
import org.apache.spark.sql.Row;
import org.apache.spark.sql.SparkSession;

import java.io.IOException;

public class Main {
    public static void main(String[] args) {
        // Creates a SparkSession
        SparkSession spark = SparkSession
                .builder()
                .appName("Main")
                .getOrCreate();

        // Load a LDA model.
        LDAModel model = DistributedLDAModel.load("/usr/project/SimpleProject/model/LDAmodel2");


        // Describe topics.
        Dataset<Row> topics = model.describeTopics(3);
        System.out.println("The topics described by their top-weighted terms:");
        topics.show(false);

        // Shows the result.
        Dataset<Row> transformed = spark.read().parquet("/usr/project/SimpleProject/model/docRepresentation.parquet");
        transformed.show(false);
        // $example off$

        spark.stop();
    }
}
