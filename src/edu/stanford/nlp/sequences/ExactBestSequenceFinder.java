package edu.stanford.nlp.sequences;

import edu.stanford.nlp.util.Pair;
import edu.stanford.nlp.util.RuntimeInterruptedException;
import edu.stanford.nlp.util.logging.Redwood;

import java.util.Arrays;


/**
 * A class capable of computing the best sequence given a SequenceModel.
 * Uses the Viterbi algorithm.
 *
 * @author Dan Klein
 * @author Teg Grenager (grenager@stanford.edu)
 */
public class ExactBestSequenceFinder implements BestSequenceFinder {

    /**
     * A logger for this class
     */
    private static final Redwood.RedwoodChannels log = Redwood.channels(ExactBestSequenceFinder.class);

    private static final boolean DEBUG = false;

    /**
     * Runs the Viterbi algorithm on the sequence model given by the TagScorer
     * in order to find the best sequence.
     *
     * @param ts The SequenceModel to be used for scoring
     * @return An array containing the int tags of the best sequence
     */
    @Override
    public int[] bestSequence(SequenceModel ts) {
        return bestSequence(ts, null).first();
    }

    private static Pair<int[], Double> bestSequence(SequenceModel ts, double[][] linearConstraints) {
        // Set up tag options
        int length = ts.length();
        int leftWindow = ts.leftWindow();
        int rightWindow = ts.rightWindow();
        int padLength = length + leftWindow + rightWindow;
        int[][] tags = new int[padLength][];
        int[] tagNum = new int[padLength];

        // INITIALIZATION DONE

        // for each position
        for (int pos = 0; pos < padLength; pos++) {
            // constraint to: only observed closed tags for the word IF word known ELSE all open tags
            tags[pos] = ts.getPossibleValues(pos);
            tagNum[pos] = tags[pos].length;
        }

        // Set up product space sizes
        int[] productSizes = new int[length];
        int curProduct = Arrays.stream(tagNum).limit(leftWindow + rightWindow).reduce((x, y) -> x * y).getAsInt();
        for (int pos = 0; pos < length; pos++) {
            curProduct *= tagNum[pos + leftWindow + rightWindow]; // shift on
            productSizes[pos] = curProduct;
            curProduct /= tagNum[pos]; // shift off
        }

        // Score all of each window's options
        int[] currentTagSequence = new int[padLength];
        double[][] windowScore = new double[length][];
        for (int pos = leftWindow; pos < leftWindow + length; pos++) {
            windowScore[pos - leftWindow] = new double[productSizes[pos - leftWindow]];
            Arrays.fill(currentTagSequence, tags[0][0]);

            for (int product = 0; product < productSizes[pos - leftWindow]; product++) {
                int p = product;
                int shift = 1;
                for (int curPos = pos + rightWindow; curPos > pos - leftWindow - 1; curPos--) {
                    currentTagSequence[curPos] = tags[curPos][p % tagNum[curPos]];
                    p /= tagNum[curPos];
                    if (curPos > pos) {
                        shift *= tagNum[curPos];
                    }
                }

                if (currentTagSequence[pos] == tags[pos][0]) {
                    // get the scores of all tags considered for the current position with respect to the whole depending tagsequence
                    // a subset of the posterior as log-probabilities
                    double[] scores = ts.scoresOf(currentTagSequence, pos);
                    // fill in the relevant windowScores
                    for (int t = 0; t < tagNum[pos]; t++) {
                        windowScore[pos - leftWindow][product + t * shift] = scores[t];
                    }
                }
            }
        }

        // Set up score and backtrace arrays
        double[][] score = new double[length][];
        int[][] trace = new int[length-1][];
        for (int pos = 0; pos < length; pos++) {
            score[pos] = new double[productSizes[pos]];
            if (pos > 0){
            trace[pos-1] = new int[productSizes[pos]];}
        }

        // Do forward Viterbi algorithm

        // loop over the classification spot

        // check for initial spot
        for (int product = 0; product < productSizes[0]; product++) {
            score[0][product] = windowScore[0][product];
        }

        // loop over the whole sentence
        for (int pos = 1; pos < length; pos++) {
            // loop over view windows
            for (int product = 0; product < productSizes[pos]; product++) {
                score[pos][product] = Double.NEGATIVE_INFINITY;
                trace[pos-1][product] = -1;
                int factor = productSizes[pos] / tagNum[pos + leftWindow + rightWindow];
                int sharedProduct = product / tagNum[pos + leftWindow + rightWindow];

                // calculate maximum
                for (int newTagNum = 0; newTagNum < tagNum[pos - 1]; newTagNum++) {
                    int predProduct = newTagNum * factor + sharedProduct;

                    // this is actually the probability of the the state.
                    // there is no transition probability considered, as there is no notion as transitions from one tag to another
                    double predScore = score[pos - 1][predProduct] + windowScore[pos][product];

                    if (predScore > score[pos][product]) {
                        score[pos][product] = predScore;
                        // this is the backpointer to the current state, therefore its correct if we incorporate the emission probability
                        trace[pos-1][product] = predProduct;
                    }
                }
            }
        }

        // Project the actual tag sequence
        double bestFinalScore = Double.NEGATIVE_INFINITY;
        // most_likely_endstate = np.argmax(log_probs[:, -1])
        // max_prob, path = log_probs[most_likely_endstate, -1], deque([most_likely_endstate])
        int bestCurrentProduct = -1;
        for (int product = 0; product < productSizes[length - 1]; product++) {
            if (score[length - 1][product] > bestFinalScore) {
                bestCurrentProduct = product;
                bestFinalScore = score[length - 1][product];
            }
        }

        int lastProduct = bestCurrentProduct;
        for (int last = padLength - 1; last >= length - 1 && last >= 0; last--) {
            currentTagSequence[last] = tags[last][lastProduct % tagNum[last]];
            lastProduct /= tagNum[last];
        }
        for (int pos = length - 2; pos >= 0; pos--) {
            int bestNextProduct = bestCurrentProduct;
            bestCurrentProduct = trace[pos][bestNextProduct];
            currentTagSequence[pos] = tags[pos][bestCurrentProduct / (productSizes[pos] / tagNum[pos])];
        }
        return new Pair<>(currentTagSequence, bestFinalScore);
    }
}
