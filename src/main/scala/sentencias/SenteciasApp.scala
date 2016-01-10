package sentencias

import org.apache.spark.SparkContext
import org.apache.spark.SparkContext._

import java.io._
import org.apache.pdfbox.pdmodel.PDDocument
import org.apache.pdfbox.util.PDFTextStripper
import org.apache.spark.mllib.linalg.Vectors
import org.apache.spark.mllib.linalg.distributed.{CoordinateMatrix, MatrixEntry}
import utils.DocumentTermMatrix._

object SenteciasApp extends App {

  def recursiveListFiles(f: File): Array[File] = {
    val these = f.listFiles
    these ++ these.filter(_.isDirectory).flatMap(recursiveListFiles)
  }

  val outputPath = "/Users/sergio/projects/proyecto-utad/output"
  val DIRECTORY = "/Users/sergio/Desktop/urbanismo"

  val f = recursiveListFiles(new File(DIRECTORY))

  val sc = new SparkContext("local", "Simple", "$SPARK_HOME", List("target/utad-1.0-SNAPSHOT.jar"))

  val documentWords = sc.parallelize(f.toSeq)

  val result = documentWords
    .filter(file => file.getName != ".DS_Store")
    .map {
      case (file) => {
        val id = file.getName
        val pdf = PDDocument.load(DIRECTORY + "/" + file.getName)
        val stripper = new PDFTextStripper()
        stripper.setStartPage(1)
        stripper.setEndPage(2)
        val content = stripper.getText(pdf)
        (id, content)
      }
    }

  val processedDocuments = documentProcessor(result, sc)
  val mat = processedDocuments._1
  val documentsRow = processedDocuments._2.foreach(x => println(x))

  val cosineSimilarity = mat.transpose().toRowMatrix().columnSimilarities()

  cosineSimilarity.entries.saveAsTextFile(outputPath + "/simsEstimate")


}

