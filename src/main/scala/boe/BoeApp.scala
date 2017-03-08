package boe

import org.apache.spark.mllib.linalg.distributed.{MatrixEntry}
import org.bson.types.ObjectId
import org.bson._
import org.jsoup.Jsoup


import utils.DocumentTermMatrix
import utils.DocumentTermMatrix._


object BoeApp extends App {


  if (args.length != 4) {
    println("wrong params: url port databaseName collection" )
    System.exit(0)
  }

  val url = args.apply(0)
  val port = args.apply(1)
  val database = args.apply(2)
  val collection = args.apply(3)

  import org.apache.spark.sql.SparkSession

  val sparkSession = SparkSession.builder()
    .master("local")
    .appName("MongoSparkConnectorIntro")
    .config("spark.mongodb.input.uri", "mongodb://"+ url +":"+ port +"/"+ database +"."+ collection)
    .config("spark.mongodb.output.uri", "mongodb://"+ url +":"+ port +"/"+ database +".matrixDistance")
    .getOrCreate()

  import com.mongodb.spark._
  import sparkSession.implicits._

  val df = MongoSpark.load(sparkSession.sparkContext)
  //df.filter(df("age") < 100).show()
  val documents = df

  //ds.map(x => (x.boeId.toString, Jsoup.parse(x.document.toString).text())).sample(false, 0.1, 0).foreach(x => println(x))



  val sc = sparkSession.sparkContext

  var result = documents.map(x => (x.get("_id").toString, Jsoup.parse(x.get("document").toString).text()))//.sample(false, 0.1, 0)//.foreach(x => println(x))

  val processedDocuments = documentProcessor(result, sc)
  val mat = processedDocuments._1
  val relationBetweenDocumentToRow = sc.broadcast(processedDocuments._2.map(_.swap).collectAsMap())

  val cosineSimilarity = mat.transpose().toRowMatrix().columnSimilarities()

  val matrix = cosineSimilarity.entries.map {
    case (MatrixEntry(i, j, v)) => {
      val insert = new Document()

      insert.append("from", new ObjectId(relationBetweenDocumentToRow.value(i)))
      insert.append("until", new ObjectId(relationBetweenDocumentToRow.value(j)))
      insert.append("value", v)

      (insert)
    }
  }

  matrix.saveToMongoDB()

}