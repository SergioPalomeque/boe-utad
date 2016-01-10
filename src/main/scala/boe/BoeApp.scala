package boe

import java.sql.{ResultSet, DriverManager}

import com.mongodb.hadoop.io.BSONWritable
import org.apache.spark.SparkContext
import org.apache.spark.SparkContext._

import org.bson.{BasicBSONObject, BSONObject}
import org.bson.types.ObjectId
import org.jsoup.Jsoup

import org.apache.spark.mllib.linalg.{Vectors}
import org.apache.spark.mllib.linalg.distributed.{CoordinateMatrix, MatrixEntry}

import org.apache.hadoop.conf.Configuration
import com.mongodb.hadoop.{MongoInputFormat, MongoOutputFormat, BSONFileInputFormat, BSONFileOutputFormat}

import utils.DocumentTermMatrix._

object BoeApp extends App {

  val url = "192.168.1.130"
  val port = 27017
  val database = "boeStorage"
  val collection = "boes"

  val mongoConfig = new Configuration()
  mongoConfig.set("mongo.input.uri", "mongodb://"+ url +":"+ port +"/"+ database +"."+ collection)

  val sc = new SparkContext("local", "Simple", "$SPARK_HOME", List("target/utad-1.0-SNAPSHOT.jar"))

  val documents = sc.newAPIHadoopRDD(
    mongoConfig,                // Configuration
    classOf[MongoInputFormat],  // InputFormat
    classOf[Object],            // Key type
    classOf[BSONObject])

  var result = documents.map(x => (x._1.toString, Jsoup.parse(x._2.get("document").toString).text()))//.sample(false, 0.1, 0)//.foreach(x => println(x))

  val processedDocuments = documentProcessor(result, sc)
  val mat = processedDocuments._1
  val relationBetweenDocumentToRow = sc.broadcast(processedDocuments._2.map(_.swap).collectAsMap())

  val cosineSimilarity = mat.transpose().toRowMatrix().columnSimilarities()

  //cosineSimilarity.entries.saveAsTextFile(outputPath + "/simsEstimate")

  val matrix = cosineSimilarity.entries.map {
    case (MatrixEntry(i, j, v)) => {
      val insert = new BasicBSONObject()

      insert.append("from", new ObjectId(relationBetweenDocumentToRow.value(i)))
      insert.append("until", new ObjectId(relationBetweenDocumentToRow.value(j)))
      insert.append("value", v)

      (null, insert)
    }
  }

  val outputConfig = new Configuration()
  outputConfig.set("mongo.output.uri", "mongodb://"+ url +":"+ port +"/"+ database +".matrixDistance")

  matrix.saveAsNewAPIHadoopFile(
    "file:///bogus",
    classOf[Object],
    classOf[BSONObject],
    classOf[MongoOutputFormat[Object, BSONObject]],
    outputConfig)

}