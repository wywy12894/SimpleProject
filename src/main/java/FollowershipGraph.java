import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.graphx.Edge;
import org.apache.spark.graphx.Graph;
import org.apache.spark.graphx.VertexRDD;
import org.apache.spark.graphx.lib.LabelPropagation;
import org.apache.spark.graphx.lib.PageRank;
import org.apache.spark.rdd.RDD;
import org.apache.spark.sql.SparkSession;
import org.apache.spark.storage.StorageLevel;
import scala.Tuple2;
import scala.reflect.ClassTag;

import java.util.List;

public class FollowershipGraph {
    public static void main(String[] args){
        // Creates a SparkSession
        SparkSession spark = SparkSession
                .builder()
                .appName("FollowershipGraph")
                .getOrCreate();

        JavaRDD<Edge<String>> edgeJavaRDD = spark.read()
                .textFile("/usr/project/SimpleProject/data/followers.txt")
                .javaRDD()
                .map(line->{
                    String[] pair = line.split(" ");
                    return new Edge<>(Integer.parseInt(pair[0].trim()), Integer.parseInt(pair[1].trim()), "follow");
                });
        RDD<Edge<String>> edgeRDD = JavaRDD.toRDD(edgeJavaRDD);


        ClassTag<String> stringTag = scala.reflect.ClassTag$.MODULE$.apply(String.class);

        Graph<String,String> followGraph = Graph.fromEdges(edgeRDD, "", StorageLevel.MEMORY_ONLY(),
                StorageLevel.MEMORY_ONLY(), stringTag, stringTag);
//        Graph<Object,String> result = LabelPropagation.run(followGraph, 10, stringTag);
        Graph<Object, Object> result = PageRank.run(followGraph, 50, 0.01, stringTag, stringTag);


        System.out.println("+++++++++++++++++++++++++++++++++++++");
        List<Edge<Object>> e = result.edges().toJavaRDD().collect();
        System.out.println(e);
        VertexRDD<Object> v = result.vertices();
        List<Tuple2<Object, Object>> sth =v.toJavaRDD().collect();
        System.out.println(sth);
        System.out.println("=====================================");

//        System.out.println("+++++++++++++++++++++++++++++++++++++");
//        List<Edge<String>> e = followGraph.edges().toJavaRDD().collect();
//        System.out.println(e);
//        e = result.edges().toJavaRDD().collect();
//        System.out.println(e);
//        VertexRDD<Object> v = result.vertices();
//        List<Tuple2<Object, Object>> sth =v.toJavaRDD().collect();
//        System.out.println(sth);
//        System.out.println("=====================================");

        spark.stop();
    }
}
