package edu.stanford.nlp.tagger.maxent

import java.io.IOException

import edu.stanford.nlp.io.PrintFile
import edu.stanford.nlp.ling.TaggedWord
import edu.stanford.nlp.tagger.io.TaggedFileRecord
import edu.stanford.nlp.util.ConfusionMatrix
import edu.stanford.nlp.util.concurrent.MulticoreWrapper
import edu.stanford.nlp.util.concurrent.ThreadsafeProcessor
import edu.stanford.nlp.util.logging.Redwood

/**
 * Tags data and can handle either data with gold-standard tags (computing
 * performance statistics) or unlabeled data.
 *
 * @author Kristina Toutanova
 * @version 1.0
 */
// TODO: can we break this class up in some way?  Perhaps we can
// spread some functionality into TestSentence and some into MaxentTagger
// TODO: at the very least, it doesn't seem to make sense to make it
// an object with state, rather than just some static methods
class TestClassifier @Throws(IOException::class)
@JvmOverloads constructor(private val maxentTagger: MaxentTagger, testFile: String = maxentTagger.config.file) {
    private var numRight: Int = 0
    private var numWrong: Int = 0
    val numWords: Int
        get() = numRight + numWrong

    private var unknownWords: Int = 0
    private var numWrongUnknown: Int = 0
    private var numCorrectSentences: Int = 0
    private var numSentences: Int = 0

    private var confusionMatrix: ConfusionMatrix<String> = ConfusionMatrix()
    private val config: TaggerConfig = maxentTagger.config
    private var writeDebug: Boolean = config.debug
    private val fileRecord = TaggedFileRecord.createRecord(config, testFile)
    private var saveRoot: String = (config.debugPrefix ?: fileRecord.filename())!!


    /**
     * Test on a file containing correct tags already. when updatePointers'ing from trees
     * TODO: Add the ability to have a second transformer to transform output back; possibly combine this method
     * with method below
     */
    @Throws(IOException::class)
    fun test() {
        var pf: PrintFile? = null
        if (writeDebug) pf = PrintFile("$saveRoot.test.debug")

        val verboseResults = config.verboseResults

        if (config.nThreads != 1) {
            val wrapper = MulticoreWrapper(config.nThreads, TestSentenceProcessor(maxentTagger))
            for (taggedSentence in fileRecord.reader()) {
                wrapper.put(taggedSentence)
                while (wrapper.peek()) {
                    processResults(wrapper.poll()!!, pf, verboseResults)
                }
            }
            wrapper.join()
            while (wrapper.peek()) {
                processResults(wrapper.poll()!!, pf, verboseResults)
            }
        } else {
            for (taggedSentence in fileRecord.reader()) {
                // TODO: Change to other tagger
                val testS = BaseSetTagger(maxentTagger)
                testS.setCorrectTags(taggedSentence)
                testS.tagSentence(taggedSentence, false)
                processResults(testS, pf, verboseResults)
            }
        }

        pf?.close()
    }

    private fun processResults(testS: BaseTagger, debugFile: PrintFile?, verboseResults: Boolean) {
        numSentences++

        testS.writeTagsAndErrors(debugFile, verboseResults)
        testS.updateConfusionMatrix(confusionMatrix)

        numWrong += testS.numWrong
        numRight += testS.numRight
        unknownWords += testS.numUnknown
        numWrongUnknown += testS.numWrongUnknown
        if (testS.numWrong == 0) {
            numCorrectSentences++
        }
        if (verboseResults) {
            log.info("Sentence number: $numSentences; length ${testS.size - 1}; " +
                    "correct: ${testS.numRight}; wrong: ${testS.numWrong}; unknown wrong: ${testS.numWrongUnknown}")
        }
    }

    private fun resultsString(maxentTagger: MaxentTagger): String {
        val output = StringBuilder()
        output.append(String.format("Model %s has xSize=%d, ySize=%d, and numFeatures=%d.%n",
                maxentTagger.config.model,
                maxentTagger.xSize,
                maxentTagger.ySize,
                maxentTagger.lambdaSolve.lambda.size))
        output.append(String.format("Results on %d sentences and %d words, of which %d were unknown.%n",
                numSentences, numRight + numWrong, unknownWords))
        output.append(String.format("Total sentences right: %d (%f%%); wrong: %d (%f%%).%n",
                numCorrectSentences, numCorrectSentences * 100.0 / numSentences,
                numSentences - numCorrectSentences,
                (numSentences - numCorrectSentences) * 100.0 / numSentences))
        output.append(String.format("Total tags right: %d (%f%%); wrong: %d (%f%%).%n",
                numRight, numRight * 100.0 / (numRight + numWrong), numWrong,
                numWrong * 100.0 / (numRight + numWrong)))

        if (unknownWords > 0) {
            output.append(String.format("Unknown words right: %d (%f%%); wrong: %d (%f%%).%n",
                    unknownWords - numWrongUnknown,
                    100.0 - numWrongUnknown * 100.0 / unknownWords,
                    numWrongUnknown, numWrongUnknown * 100.0 / unknownWords))
        }

        return output.toString()
    }

    fun printModelAndAccuracy(maxentTagger: MaxentTagger) {
        // print the output all at once so that multiple threads don't clobber each other's output
        log.info(resultsString(maxentTagger))
    }

    internal class TestSentenceProcessor(private var maxentTagger: MaxentTagger) : ThreadsafeProcessor<List<TaggedWord>, BaseTagger> {
        override fun process(taggedSentence: List<TaggedWord>): BaseTagger {
            val testS = BaseTagger(maxentTagger)
            testS.setCorrectTags(taggedSentence)
            testS.tagSentence(taggedSentence, false)
            return testS
        }

        override fun newInstance(): ThreadsafeProcessor<List<TaggedWord>, BaseTagger> {
            // MaxentTagger is threadsafe
            return this
        }
    }

    companion object {
        private val log = Redwood.channels(TestClassifier::class.java)
    }
}
