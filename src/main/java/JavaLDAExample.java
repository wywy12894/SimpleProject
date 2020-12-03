// $example on$
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.ml.clustering.LDA;
import org.apache.spark.ml.clustering.LDAModel;
import org.apache.spark.sql.Dataset;
import org.apache.spark.sql.Row;
import org.apache.spark.sql.SparkSession;
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
        input = transformed.join(input, "label");
        input.show(false);



//        try {
//            model.write().overwrite().save("/usr/project/SimpleProject/model/LDAmodel2");
//            transformed.write().parquet("/usr/project/SimpleProject/model/docRepresentation.parquet");
//        } catch (IOException e) {
//            e.printStackTrace();
//        }

        spark.stop();
    }
}
