package edu.stanford.nlp.sequences

import edu.stanford.nlp.util.Pair
import edu.stanford.nlp.util.logging.Redwood
import java.lang.Double.NEGATIVE_INFINITY


/**
 * A class capable of computing the best sequence given a SequenceModel.
 * Uses the Viterbi algorithm.
 *
 * @author Dan Klein
 * @author Teg Grenager (grenager@stanford.edu)
 */
class ExactBestSequenceFinder : BestSequenceFinder {

    /**
     * Runs the Viterbi algorithm on the sequence model given by the TagScorer
     * in order to find the best sequence.
     *
     * @param ts The SequenceModel to be used for scoring
     * @return An array containing the int tags of the best sequence
     */
    override fun bestSequence(ts: SequenceModel): IntArray {
        // Set up tag options
        val length = ts.length()
        val leftWindow = ts.leftWindow()
        val rightWindow = ts.rightWindow()
        val padLength = length + leftWindow + rightWindow

        // constraint to: only observed closed tags for the word IF word known ELSE all open tags
        val tags = Array<IntArray>(padLength) { ts.getPossibleValues(it) }
        val tagNum = IntArray(padLength) { tags[it].size }

        // Set up product space sizes
        val productSizes = IntArray(length)
        var curProduct = tagNum.take(leftWindow + rightWindow).reduce { x, y -> x * y }
        for (pos in 0 until length) {
            curProduct *= tagNum[pos + leftWindow + rightWindow] // shift on
            productSizes[pos] = curProduct
            curProduct /= tagNum[pos] // shift off
        }

        // Score all of each window's options
        val currentTagSequence = IntArray(padLength)
        val windowScore = Array(length) { DoubleArray(productSizes[it]) }
        for (pos in leftWindow until leftWindow + length) {
            currentTagSequence.fill(tags[0][0])

            for (product in 0 until productSizes[pos - leftWindow]) {
                var p = product
                var shift = 1
                for (curPos in pos + rightWindow downTo pos - leftWindow - 1 + 1) {
                    currentTagSequence[curPos] = tags[curPos][p % tagNum[curPos]]
                    p /= tagNum[curPos]
                    if (curPos > pos) {
                        shift *= tagNum[curPos]
                    }
                }

                if (currentTagSequence[pos] == tags[pos][0]) {
                    // get the scores of all tags considered for the current position with respect to the whole depending tagsequence
                    // a subset of the posterior as log-probabilities
                    val scores = ts.scoresOf(currentTagSequence, pos)
                    // fill in the relevant windowScores
                    for (t in 0 until tagNum[pos]) {
                        windowScore[pos - leftWindow][product + t * shift] = scores[t]
                    }
                }
            }
        }

        // Set up score and backtrace arrays
        val score = Array(length) { DoubleArray(productSizes[it]) }
        val trace = Array(length - 1) { IntArray(productSizes[it + 1]) }


        // Do forward Viterbi algorithm


        // check for initial spot
        if (productSizes[0] >= 0) System.arraycopy(windowScore[0], 0, score[0], 0, productSizes[0])


        // loop over the classification spot (positions in sentence)
        for (pos in 1 until length) {
            // loop over view windows
            for (product in 0 until productSizes[pos]) {
                score[pos][product] = NEGATIVE_INFINITY
                trace[pos - 1][product] = -1
                val factor = productSizes[pos] / tagNum[pos + leftWindow + rightWindow]
                val sharedProduct = product / tagNum[pos + leftWindow + rightWindow]

                // calculate maximum
                for (newTagNum in 0 until tagNum[pos - 1]) {
                    val predProduct = newTagNum * factor + sharedProduct

                    // this is actually the probability of the the state.
                    // there is no transition probability considered, as there is no notion as transitions from one tag to another
                    val predScore = score[pos - 1][predProduct] + windowScore[pos][product]

                    if (predScore > score[pos][product]) {
                        score[pos][product] = predScore
                        // this is the backpointer to the current state, therefore its correct if we incorporate the emission probability
                        trace[pos - 1][product] = predProduct
                    }
                }
            }
        }

        // Project the actual tag sequence
        var bestFinalScore = java.lang.Double.NEGATIVE_INFINITY
        // most_likely_endstate = np.argmax(log_probs[:, -1])
        // max_prob, path = log_probs[most_likely_endstate, -1], deque([most_likely_endstate])
        var bestCurrentProduct = -1
        for (product in 0 until productSizes[length - 1]) {
            if (score[length - 1][product] > bestFinalScore) {
                bestCurrentProduct = product
                bestFinalScore = score[length - 1][product]
            }
        }

        var lastProduct = bestCurrentProduct
        var last = padLength - 1
        while (last >= length - 1 && last >= 0) {
            currentTagSequence[last] = tags[last][lastProduct % tagNum[last]]
            lastProduct /= tagNum[last]
            last--
        }
        for (pos in length - 2 downTo 0) {
            val bestNextProduct = bestCurrentProduct
            bestCurrentProduct = trace[pos][bestNextProduct]
            currentTagSequence[pos] = tags[pos][bestCurrentProduct / (productSizes[pos] / tagNum[pos])]
        }
        return currentTagSequence
    }
}
