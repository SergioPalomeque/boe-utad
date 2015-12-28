package boe

import java.sql.{ResultSet, DriverManager}
import java.text.Normalizer
import java.util.regex.Pattern

import scala.collection.mutable.HashMap

import org.apache.spark.SparkContext
import org.apache.spark.SparkContext._

import org.apache.spark.rdd.JdbcRDD
import org.jsoup.Jsoup

import org.apache.spark.mllib.linalg.{Vectors, Vector}
import org.apache.spark.mllib.linalg.distributed.{RowMatrix, MatrixEntry}
import org.apache.spark.mllib.clustering.{KMeans, KMeansModel}


object BoeApp extends App {

  /*
   * Pametrizar palabras validas. Apartir de 3
   * Limpia el texto, minúsculas, sin tildes ni signos de puntuación como coma, punto. Los caracteres alfanuméricos y números se eliminan
   * Filtrar palabras que no esten el la lista stopwords
   * Crear mega matriz. La fila es el documento. La columna las palabras del documento. El valor sera el numero de veces que esta ese documento
   */

  var outputPath = "/Users/sergio/projects/proyecto-utad/output"

  val url = "192.168.1.42"
  val port = "5432"
  val database = "boe"
  val user = "sergio"
  val password = "postgres"

  val stopWordsList = List("algún","alguna","algunas","alguno","algunos","ambos","ampleamos","ante","antes","aquel","aquellas","aquellos","aqui","arriba","atras","bajo","bastante","bien","cada","cierta","ciertas","cierto","ciertos","como","con","conseguimos","conseguir","consigo","consigue","consiguen","consigues","cual","cuando","dentro","desde","donde","dos","el","ellas","ellos","empleais","emplean","emplear","empleas","empleo","en","encima","entonces","entre","era","eramos","eran","eras","eres","es","esta","estaba","estado","estais","estamos","estan","estoy","fin","fue","fueron","fui","fuimos","gueno","ha","hace","haceis","hacemos","hacen","hacer","haces","hago","incluso","intenta","intentais","intentamos","intentan","intentar","intentas","intento","ir","la","largo","las","lo","los","mientras","mio","modo","muchos","muy","nos","nosotros","otro","para","pero","podeis","podemos","poder","podria","podriais","podriamos","podrian","podrias","por","que","porque","primero","puede","pueden","puedo","quien","sabe","sabeis","sabemos","saben","saber","sabes","ser","si","siendo","sin","sobre","sois","solamente","solo","somos","soy","su","sus","tambie","teneis","tenemos","tener","tengo","tiempo","tiene","tienen","todo","trabaja","trabajais","trabajamos","trabajan","trabajar","trabajas","trabajo","tras","tuyo","ultimo","un","una","unas","uno","unos","usa","usais","usamos","usan","usar","usas","uso","va","vais","valor","vamos","van","vaya","verdad","verdadera","verdadero","vosotras","vosotros","voy","yo")

  val sc = new SparkContext("local", "Simple", "$SPARK_HOME", List("target/utad-1.0-SNAPSHOT.jar"))

  def createConnection () = {
    DriverManager.getConnection("jdbc:postgresql://"+ url +":"+ port +"/"+ database +"?user="+ user +"&password="+ password)
  }

  def removeAccent(str: String): String = {
    val nfdNormalizedString = Normalizer.normalize(str, Normalizer.Form.NFD)
    val pattern = Pattern.compile("\\p{InCombiningDiacriticalMarks}+")
    pattern.matcher(nfdNormalizedString).replaceAll("")
  }

  def extractValues (r: ResultSet): BoeDocument = {
    var doc = Jsoup.parse(r.getString("texto")).text()
    doc = removeAccent(doc)
    val textWords = doc.replaceAll("[^a-zA-Z ]", "").toLowerCase() //.split("\\s+")
    new BoeDocument(r.getString("identificador"), textWords)
  }

  val query = "SELECT identificador, texto FROM boe_analisis_documento LIMIT ? OFFSET ?"
  val data = new JdbcRDD(sc, createConnection, query, lowerBound = 1, upperBound = 50, numPartitions = 2, mapRow = extractValues)
  var wordsList = new HashMap[String, Int]()

  def splitLinesInWords (x: BoeDocument) = {
    x.getText.split(" ").map(y => (x.getId, y))
  }

  def cleanText (x: (String, String)) = {
    {x._2.length >= 3 && !stopWordsList.contains(x._2)}
  }

  val documentWords = data.flatMap(splitLinesInWords)
  val filterWords = documentWords.filter(cleanText)

  val totalWords = filterWords.map(x => x._2).distinct().collect()
  println("********** Total words ***********")
  println("* " + totalWords.length + " *")
  println("*********************************")
  val bWords = sc.broadcast(totalWords)

  val zero = (Array.empty[Int], Array.empty[Double])

  var wordCountByDocument = filterWords
    .map(x => ((x._1, x._2), 1))
    .reduceByKey((acc, value) => acc + value)
    .map(x => (x._1._1, (x._1._2, x._2)))
    .aggregateByKey(zero)((acc, curr) => {
        val index = bWords.value.indexOf(curr._1)
        var indexes = acc._1
        indexes :+= index
        var values = acc._2
        values :+= curr._2.toDouble
        (indexes, values)
      },
      (left, right) => {
        (left._1 ++ right._1, left._2 ++ right._2)
      })
    .map(x => {
      val indexes = x._2._1
      val values = x._2._2
      Vectors.sparse(bWords.value.length, indexes, values)
    })
    //.foreach(x => println(x))

  val mat = new RowMatrix(wordCountByDocument)

  println("numRows: " + mat.numRows())

  val simsPerfect = mat.columnSimilarities()
  val simsEstimate = mat.columnSimilarities(0.8)

  simsEstimate.entries.saveAsTextFile(outputPath + "/simsEstimate.txt")
  wordCountByDocument.saveAsTextFile(outputPath + "/wordCountByDocument.txt")
  //println("Pairwise similarities are: " + simsPerfect.entries.collect.mkString(", "))

  //println("Estimated pairwise similarities are: " +     simsEstimate.entries.collect.mkString(", "))
}