package utils

import java.text.Normalizer
import java.util.regex.Pattern

import org.apache.spark.SparkContext
import org.apache.spark.mllib.linalg.Vectors
import org.apache.spark.mllib.linalg.distributed.{CoordinateMatrix, MatrixEntry}
import org.apache.spark.rdd.RDD
object DocumentTermMatrix {

  val stopWordsList = List("algun","alguna","algunas","alguno","algunos","ambos","ampleamos","ante","antes","aquel","aquellas","aquellos","aqui","arriba","atras","bajo","bastante","bien","cada","cierta","ciertas","cierto","ciertos","como","con","conseguimos","conseguir","consigo","consigue","consiguen","consigues","cual","cuando","dentro","desde","donde","dos","el","ellas","ellos","empleais","emplean","emplear","empleas","empleo","en","encima","entonces","entre","era","eramos","eran","eras","eres","es","esta","estaba","estado","estais","estamos","estan","estoy","fin","fue","fueron","fui","fuimos","gueno","ha","hace","haceis","hacemos","hacen","hacer","haces","hago","incluso","intenta","intentais","intentamos","intentan","intentar","intentas","intento","ir","la","largo","las","lo","los","mientras","mio","modo","muchos","muy","nos","nosotros","nosotras","otro","para","pero","podeis","podemos","poder","podria","podriais","podriamos","podrian","podrias","por","que","porque","primero","puede","pueden","puedo","quien","sabe","sabeis","sabemos","saben","saber","sabes","ser","si","siendo","sin","sobre","sois","solamente","solo","somos","soy","su","sus","tambien","teneis","tenemos","tener","tengo","tiempo","tiene","tienen","todo","trabaja","trabajais","trabajamos","trabajan","trabajar","trabajas","trabajo","tras","tuyo","tuya","ultimo","un","una","unas","uno","unos","usa","usais","usamos","usan","usar","usas","uso","va","vais","valor","vamos","van","vaya","verdad","verdadera","verdadero","vosotras","vosotros","voy","yo","ninguno","ninguna","ningunos","ningunas","tampoco","tanto","tantos","tanta","tantas","demasiado","demasiada","demasiados","demasiadas","quienquiera","cabe","casi","menos","mia","mias","usted","ustedes","hasta")

  def removeAccent(str: String): String = {
    val nfdNormalizedString = Normalizer.normalize(str, Normalizer.Form.NFD)
    val pattern = Pattern.compile("\\p{InCombiningDiacriticalMarks}+")
    pattern.matcher(nfdNormalizedString).replaceAll("")
  }

  def extractValues (r: (String, String)): (String, String) = {
    val doc = removeAccent(r._2)
    val textWords = doc.replaceAll("[\\t\\n\\r]+", " ").replaceAll("[^a-zA-Z ]", "").toLowerCase()
    (r._1, textWords)
  }


  def splitLinesInWords (x: (String, String)) = {
    x._2.split(" ").map(y => (x._1, y))
  }

  def cleanText (x: (String, String)) = {
    {x._2.length >= 3 && !stopWordsList.contains(x._2)}
  }

  def documentProcessor (documents: RDD[(String, String)], sc: SparkContext): (CoordinateMatrix, RDD[(String, Long)]) = {
    val documentsTotalNumber = documents.count()
    val filterWords = documents
      .map(x => extractValues(x))
      .filter(x => x._2.length > 0)
      .flatMap(splitLinesInWords)
      .filter(cleanText)
    filterWords.cache()

    val totalWords = filterWords.map(x => x._2).distinct().collect()
    println("*** Total words ***")
    println(totalWords.length)
    println("*******************")

    println("*** Total documents ***")
    println(documentsTotalNumber)
    println("***********************")

    val bWords = sc.broadcast(totalWords.zipWithIndex.toMap.map(_.swap))

    val zero = (Array.empty[Int], Array.empty[Double])

    val docTermFreqs = filterWords
      .map{
        case(id, word) => ((id, word), 1)
      }
      .reduceByKey((acc, value) => acc + value)
    docTermFreqs.cache()

    val wordCountVectorByDocument = docTermFreqs
      .map{
        case ((id, word), count) => (id, (word, count))
      }
      .aggregateByKey(zero)((acc, curr) => {
        val index = bWords.value.find(_._2 == curr._1).head._1
        var indexes = acc._1
        indexes :+= index
        var values = acc._2
        values :+= curr._2.toDouble
        (indexes, values)
      },
        (left, right) => {
          (left._1 ++ right._1, left._2 ++ right._2)
        })
      .map{
      case (id, (indices, values)) => (id, Vectors.sparse(bWords.value.size, indices, values))
    }

    val tf = wordCountVectorByDocument
      .map{
        case (id, vector) => (id, vector, vector.toSparse.values.sum)
      }
      .map {
        case (id, vector, wordsCounter) => {
          val indices = vector.toSparse.indices
          val values = vector.toSparse.values.transform(x => x/wordsCounter)
          (id, Vectors.sparse(bWords.value.size, indices, values.toArray))
        }
      }
    tf.cache()

    // numero documentos donde una palabra se encuentra
    val idf = docTermFreqs
      .map {
        case ((id, word), count) => (word, 1)
      }
      .reduceByKey((acc, value) => acc + value)
      .map {
        case (id, count) => (id, 1 + Math.log(documentsTotalNumber/count))
      }

    val bIdf = sc.broadcast(idf.collectAsMap())

    val tfIdf = tf
      .map {
        case (id, tfVector) => {
          val vector = tfVector.toSparse.values
          val indices = tfVector.toSparse.indices
          val values = new Array[Double](indices.length)
          for (i <- 0 until indices.length) {
            val word = bWords.value(indices.apply(i))
            values(i) = vector.apply(i) * bIdf.value(word)
          }
          (id, Vectors.sparse(bWords.value.size, indices, values))
        }
      }

    val tfIdfWithId = tfIdf.zipWithUniqueId()
    tfIdfWithId.cache()

    val documentRow = tfIdfWithId
      .map {
        case ((id, vector), index) => {
          (id, index)
        }
      }

    val matrix = tfIdfWithId
      .map {
        case ((id, vector), index) => {
          val indices = vector.toSparse.indices
          var l = List[MatrixEntry]()
          for(j <- 0 until indices.length) {
            l = MatrixEntry(index, vector.toSparse.indices.apply(j), vector.toSparse.values.apply(j)) :: l
          }
          l
        }
      }
      .flatMap(x => x)

    (new CoordinateMatrix(matrix), documentRow)
  }
}
