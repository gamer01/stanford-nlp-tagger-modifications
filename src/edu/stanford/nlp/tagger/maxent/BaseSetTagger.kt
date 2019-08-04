package edu.stanford.nlp.tagger.maxent

import edu.stanford.nlp.io.PrintFile

class BaseSetTagger(maxentTagger: MaxentTagger?) : BaseTagger(maxentTagger) {

    internal override fun writeTagsAndErrors(pf: PrintFile?, verboseResults: Boolean) {
        super.writeTagsAndErrors(pf, verboseResults)
        // call ubop for the whole sequence
        val finalTagSets = deriveTagSets(::genSingletons,false)
        println(finalTagSets.joinToString())

        //write stuff to csv

    }

    private fun deriveTagSets(setpredictor: (scores: DoubleArray, tags: Array<String>) -> Set<String> = ::genSingletons,
                              constraintTags: Boolean = true): List<Set<String>> {
        // fill left and right window with NA tags and convert tags to tagindices
        val sequence = (List(leftWindow()) { naTag } + finalTags + List(rightWindow()) { naTag }).map { maxentTagger.tags.getIndex(it) }.toIntArray()


        // skip end of sentence tag
        val result = (0 until size - 1).map { pos ->
            // in each position we call UBOP to derive the set-valued prediction
            if (constraintTags) {
                val scores = scoresOf(sequence, pos + leftWindow())
                val tags = getPossibleTagsAsString(pos + leftWindow())
                setpredictor(scores, tags)
            } else {
                history.updatePointers(0, size - 1, pos )
                setHistory(pos+leftWindow(), sequence)
                val scores = allScores
                val tags = maxentTagger.tags.tagSet().sortedBy { maxentTagger.tags.getIndex(it) }.toTypedArray()
                setpredictor(scores, tags)
            }
        }

        return result
    }

    private fun genSingletons(scores: DoubleArray, filteredTags: Array<String>) =
            setOf("${scores.size}:" + scores.indices.maxBy { scores[it] }?.let { filteredTags[it] })


}