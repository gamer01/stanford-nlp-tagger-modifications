package edu.stanford.nlp.tagger.maxent

import edu.stanford.nlp.io.PrintFile

class BaseSetTagger(maxentTagger: MaxentTagger?) : BaseTagger(maxentTagger) {
    companion object {
        private var n_sent = 1
    }


    internal override fun writeTagsAndErrors(pf: PrintFile?, verboseResults: Boolean) {
        super.writeTagsAndErrors(null, verboseResults)
        // call ubop for the whole sequence
        //val finalTagSets = deriveTagSets(::genSingletons, false)
        //println(finalTagSets.joinToString())

        if (pf == null)
            return

        //write stuff to csv
        val sequence = (List(leftWindow()) { naTag } + finalTags + List(rightWindow()) { naTag }).map { maxentTagger.tags.indexOf(it) }.toIntArray()

        // skip end of sentence tag
        for (pos in 0 until size - 1) {
            //word; sentenceID; isunknown; truelabel; label posterior; constrained tags;
            val word = sent[pos]
            val data = arrayOf(
                    word,
                    n_sent.toString(),
                    maxentTagger.dict.isUnknown(word).toString(),
                    correctTags[pos],
                    finalTags[pos],
                    scoresOf(sequence, pos + leftWindow(), false).joinToString(prefix = "[", postfix = "]"),
                    getPossibleTagsAsString(pos + leftWindow()).joinToString(prefix = "[", postfix = "]")
            )
            pf.println(data.joinToString(separator = ";", transform = { "\"$it\"" }))
            //if (!getPossibleTagsAsString(pos + leftWindow()).contains(correctTags[pos]))
            //    println("forced misclassification: ${finalTags[pos]}; ${getPossibleTagsAsString(pos + leftWindow()).joinToString(prefix = "[", postfix = "]")}")
        }

        n_sent++
    }

    private fun deriveTagSets(setpredictor: (scores: DoubleArray, tags: Array<String>) -> Set<String> = ::genSingletons,
                              constraintTags: Boolean = true): List<Set<String>> {
        // fill left and right window with NA tags and convert tags to tagindices
        val sequence = (List(leftWindow()) { naTag } + finalTags + List(rightWindow()) { naTag }).map { maxentTagger.tags.indexOf(it) }.toIntArray()


        // skip end of sentence tag
        val result = (0 until size - 1).map { pos ->
            // in each position we call the set-valued predictor
            if (constraintTags) {
                val scores = scoresOf(sequence, pos + leftWindow())
                val tags = getPossibleTagsAsString(pos + leftWindow())
                setpredictor(scores, tags)
            } else {
                // similar to above, but we do not constraint the scores.
                val scores = scoresOf(sequence, pos + leftWindow(), false)
                val tags = maxentTagger.tags.tagSet().sortedBy { maxentTagger.tags.indexOf(it) }.toTypedArray()
                setpredictor(scores, tags)
            }
        }

        return result
    }

    private fun genSingletons(scores: DoubleArray, filteredTags: Array<String>) =
            setOf("${scores.size}:" + scores.indices.maxBy { scores[it] }?.let { filteredTags[it] })
}