package edu.stanford.nlp.tagger.maxent

import edu.stanford.nlp.io.PrintFile

class BaseSetTagger(maxentTagger: MaxentTagger?) : BaseTagger(maxentTagger) {

    internal override fun writeTagsAndErrors(pf: PrintFile?, verboseResults: Boolean) {
        super.writeTagsAndErrors(pf, verboseResults)
        // call ubop for the whole sequence
        //val FinalTagSets = deriveTagSets()

        //write stuff to csv

    }

    private fun deriveTagSets(): List<Set<String>> {
        // fill left and right window with NA tags and convert tags to tagindices
        val tags = (List(leftWindow()) { naTag } + finalTags + List(rightWindow()) { naTag }).map { maxentTagger.tags.getIndex(it) }
        val windowWidth = leftWindow() + rightWindow()

        initializeScorer()
        (0 until size).map { pos: Int ->
            // in each position we call UBOP to derive the set-valued prediction
            val scores = scoresOf(tags.slice(pos..windowWidth).toIntArray(), pos + leftWindow())
            setOf<String>("")
        }

        return listOf(setOf(" "))
    }


}