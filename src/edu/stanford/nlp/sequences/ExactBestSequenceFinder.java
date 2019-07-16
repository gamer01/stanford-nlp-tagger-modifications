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
        if (linearConstraints != null && linearConstraints.length != padLength)
            throw new RuntimeException("linearConstraints.length (" + linearConstraints.length + ") does not match padLength (" + padLength + ") of SequenceModel" + ", length==" + length + ", leftW=" + leftWindow + ", rightW=" + rightWindow);
        int[][] tags = new int[padLength][];
        int[] tagNum = new int[padLength];
        if (DEBUG) {
            log.info("Doing bestSequence length " + length + "; leftWin " + leftWindow + "; rightWin " + rightWindow + "; padLength " + padLength);
        }

        // INITIALIZATION DONE

        // for each position
        for (int pos = 0; pos < padLength; pos++) {
            if (Thread.interrupted()) {  // Allow interrupting
                throw new RuntimeInterruptedException();
            }
            // constraint to: only observed closed tags for the word IF word known ELSE all open tags
            tags[pos] = ts.getPossibleValues(pos);
            tagNum[pos] = tags[pos].length;
            if (DEBUG) {
                log.info("There are " + tagNum[pos] + " values at position " + pos + ": " + Arrays.toString(tags[pos]));
            }
        }

        int[] tempTags = new int[padLength];

        // Set up product space sizes
        int[] productSizes = new int[padLength];


        int curProduct = Arrays.stream(tagNum).limit(leftWindow + rightWindow).reduce((x, y) -> x * y).getAsInt();

        for (int pos = leftWindow + rightWindow; pos < padLength; pos++) {
            if (Thread.interrupted()) {  // Allow interrupting
                throw new RuntimeInterruptedException();
            }
            curProduct *= tagNum[pos]; // shift on
            productSizes[pos - rightWindow] = curProduct;
            curProduct /= tagNum[pos - leftWindow - rightWindow]; // shift off
        }

        // Score all of each window's options
        double[][] windowScore = new double[padLength][];
        for (int pos = leftWindow; pos < leftWindow + length; pos++) {
            windowScore[pos] = new double[productSizes[pos]];
            Arrays.fill(tempTags, tags[0][0]);

            for (int product = 0; product < productSizes[pos]; product++) {
                int p = product;
                int shift = 1;
                for (int curPos = pos + rightWindow; curPos >= pos - leftWindow; curPos--) {
                    tempTags[curPos] = tags[curPos][p % tagNum[curPos]];
                    p /= tagNum[curPos];
                    if (curPos > pos) {
                        shift *= tagNum[curPos];
                    }
                }

                if (tempTags[pos] == tags[pos][0]) {
                    // get all tags at once
                    double[] scores = ts.scoresOf(tempTags, pos);
                    // fill in the relevant windowScores
                    for (int t = 0; t < tagNum[pos]; t++) {
                        windowScore[pos][product + t * shift] = scores[t];
                    }
                }
            }
        }

        // Set up score and backtrace arrays
        double[][] score = new double[padLength][];
        int[][] trace = new int[padLength][];
        for (int pos = 0; pos < padLength; pos++) {
            score[pos] = new double[productSizes[pos]];
            trace[pos] = new int[productSizes[pos]];
        }

        // Do forward Viterbi algorithm

        // loop over the classification spot
        for (int pos = leftWindow; pos < length + leftWindow; pos++) {
            // loop over window product types
            for (int product = 0; product < productSizes[pos]; product++) {
                // check for initial spot
                if (pos == leftWindow) {
                    // no predecessor type
                    score[pos][product] = windowScore[pos][product];
                    trace[pos][product] = -1;
                } else {
                    // loop over possible predecessor types
                    score[pos][product] = Double.NEGATIVE_INFINITY;
                    trace[pos][product] = -1;
                    int sharedProduct = product / tagNum[pos + rightWindow];
                    int factor = productSizes[pos] / tagNum[pos + rightWindow];
                    for (int newTagNum = 0; newTagNum < tagNum[pos - leftWindow - 1]; newTagNum++) {
                        int predProduct = newTagNum * factor + sharedProduct;
                        double predScore = score[pos - 1][predProduct] + windowScore[pos][product];

                        if (predScore > score[pos][product]) {
                            score[pos][product] = predScore;
                            trace[pos][product] = predProduct;
                        }
                    }
                }
            }
        }

        // Project the actual tag sequence
        double bestFinalScore = Double.NEGATIVE_INFINITY;
        int bestCurrentProduct = -1;
        for (int product = 0; product < productSizes[leftWindow + length - 1]; product++) {
            if (score[leftWindow + length - 1][product] > bestFinalScore) {
                bestCurrentProduct = product;
                bestFinalScore = score[leftWindow + length - 1][product];
            }
        }
        int lastProduct = bestCurrentProduct;
        for (int last = padLength - 1; last >= length - 1 && last >= 0; last--) {
            tempTags[last] = tags[last][lastProduct % tagNum[last]];
            lastProduct /= tagNum[last];
        }
        for (int pos = leftWindow + length - 2; pos >= leftWindow; pos--) {
            int bestNextProduct = bestCurrentProduct;
            bestCurrentProduct = trace[pos + 1][bestNextProduct];
            tempTags[pos - leftWindow] = tags[pos - leftWindow][bestCurrentProduct / (productSizes[pos] / tagNum[pos - leftWindow])];
        }
        return new Pair<>(tempTags, bestFinalScore);
    }
}
