// MaxentTagger -- StanfordMaxEnt, A Maximum Entropy Toolkit
// Copyright (c) 2002-2016 Leland Stanford Junior University

// This program is free software; you can redistribute it and/or
// modify it under the terms of the GNU General Public License
// as published by the Free Software Foundation; either version 2
// of the License, or (at your option) any later version.

// This program is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
// GNU General Public License for more details.

// You should have received a copy of the GNU General Public License
// along with this program; if not, write to the Free Software
// Foundation, Inc., 59 Temple Place - Suite 330, Boston, MA  02111-1307, USA.

// For more information, bug reports, fixes, contact:
// Christopher Manning
// Dept of Computer Science, Gates 2A
// Stanford CA 94305-9020
// USA
// Support/Questions: stanford-nlp on SO or java-nlp-user@lists.stanford.edu
// Licensing: java-nlp-support@lists.stanford.edu
// http://nlp.stanford.edu/software/tagger.html

package edu.stanford.nlp.tagger.maxent;

import edu.stanford.nlp.io.PrintFile;
import edu.stanford.nlp.ling.*;
import edu.stanford.nlp.math.ArrayMath;
import edu.stanford.nlp.sequences.BestSequenceFinder;
import edu.stanford.nlp.sequences.ExactBestSequenceFinder;
import edu.stanford.nlp.sequences.SequenceModel;
import edu.stanford.nlp.tagger.common.Tagger;
import edu.stanford.nlp.util.*;
import edu.stanford.nlp.util.logging.Redwood;
import org.jetbrains.annotations.Contract;

import java.io.OutputStreamWriter;
import java.io.PrintWriter;
import java.io.StringWriter;
import java.io.UnsupportedEncodingException;
import java.util.*;
import java.util.stream.Stream;


/**
 * @author Kristina Toutanova
 * @author Michel Galley
 * @version 1.0
 */
public class BaseTagger implements SequenceModel {

    /**
     * A logger for this class
     */
    protected static final Redwood.RedwoodChannels log = Redwood.channels(BaseTagger.class);

    protected static final String naTag = "EMPTY";
    protected static final String[] naTagArr = {naTag};
    protected static final boolean DBG = false;

    protected final String tagSeparator;
    protected final String encoding;
    protected final PairsHolder pairs = new PairsHolder();
    protected List<String> sent;
    // is only set inside "tagSentence" and only if the "reuseTags" flag is set true.
    private List<String> originalTags;
    // origWords is only set when run with a list of HasWords; when run
    // with a list of strings, this will be null
    private List<HasWord> origWords;
    protected int size; // TODO this always has the value of sent.size(). Remove it? [cdm 2008]
    // protected double[][][] probabilities;
    protected String[] correctTags;
    String[] finalTags;
    int numRight;
    int numWrong;
    int numUnknown;
    int numWrongUnknown;

    private volatile History history;
    private volatile Map<String, double[]> localScores = Generics.newHashMap();
    private volatile double[][] localContextScores;

    protected final MaxentTagger maxentTagger;

    public BaseTagger(MaxentTagger maxentTagger) {
        assert (maxentTagger != null);
        assert (maxentTagger.getLambdaSolve() != null);
        this.maxentTagger = maxentTagger;
        if (maxentTagger.config != null) {
            tagSeparator = maxentTagger.config.getTagSeparator();
            encoding = maxentTagger.config.getEncoding();
        } else {
            tagSeparator = TaggerConfig.getDefaultTagSeparator();
            encoding = "utf-8";
        }
        history = new History(pairs, maxentTagger.extractors);
    }

    void setCorrectTags(List<? extends HasTag> sentence) {
        correctTags = sentence.stream().map(HasTag::tag).toArray(String[]::new);
    }

    /**
     * Tags the sentence s by running maxent model.  Returns a sentence (List) of
     * TaggedWord objects.
     *
     * @param s Input sentence (List).  This isn't changed.
     * @return Tagged sentence
     */
    public ArrayList<TaggedWord> tagSentence(List<? extends HasWord> s,
                                             boolean reuseTags) {
        this.origWords = new ArrayList<>(s);
        int sz = s.size();
        this.sent = new ArrayList<>(sz + 1);
        for (HasWord value1 : s) {
            if (maxentTagger.wordFunction != null) {
                sent.add(maxentTagger.wordFunction.apply(value1.word()));
            } else {
                sent.add(value1.word());
            }
        }
        sent.add(Tagger.EOS_WORD);
        if (reuseTags) {
            this.originalTags = new ArrayList<>(sz + 1);
            for (HasWord value : s) {
                if (value instanceof HasTag) {
                    originalTags.add(((HasTag) value).tag());
                } else {
                    originalTags.add(null);
                }
            }
            originalTags.add(Tagger.EOS_TAG);
        }
        size = sz + 1;
        init();
        ArrayList<TaggedWord> result = testTagInference();
        if (maxentTagger.wordFunction != null) {
            for (int j = 0; j < sz; ++j) {
                result.get(j).setWord(s.get(j).word());
            }
        }
        return result;
    }

    protected void init() {
        //the eos are assumed already there
        localContextScores = new double[size][];
        numUnknown += sent.stream().filter(maxentTagger.dict::isUnknown).count();
    }


    private ArrayList<TaggedWord> getTaggedSentence() {
        final boolean hasOffset;
        hasOffset = origWords != null && !origWords.isEmpty() && (origWords.get(0) instanceof HasOffset);
        ArrayList<TaggedWord> taggedSentence = new ArrayList<>();
        for (int j = 0; j < size - 1; j++) {
            String tag = finalTags[j];
            TaggedWord w = new TaggedWord(sent.get(j), tag);
            if (hasOffset) {
                HasOffset offset = (HasOffset) origWords.get(j);
                w.setBeginPosition(offset.beginPosition());
                w.setEndPosition(offset.endPosition());
            }
            taggedSentence.add(w);
        }
        return taggedSentence;
    }

    @Contract(value = "!null -> !null", pure = true)
    static String toNice(String s) {
        return Objects.requireNonNullElse(s, naTag);
    }


    /**
     * Write the tagging and note any errors (if pf != null) and accumulate
     * global statistics.
     *
     * @param pf File to write tagged output to (can be null, then no output;
     *           at present it is non-null iff the debug property is set)
     */
    void writeTagsAndErrors(PrintFile pf, boolean verboseResults) {
        StringWriter sw = new StringWriter(200);
        for (int i = 0; i < correctTags.length; i++) {
            sw.write(toNice(sent.get(i)));
            sw.write(tagSeparator);
            sw.write(finalTags[i]);
            sw.write(' ');
            if (pf != null) {
                pf.print(toNice(sent.get(i)));
                pf.print(tagSeparator);
                pf.print(finalTags[i]);
            }
            if ((correctTags[i]).equals(finalTags[i])) {
                numRight++;
            } else {
                numWrong++;
                if (pf != null) pf.print('|' + correctTags[i]);
                if (verboseResults) {
                    log.info((maxentTagger.dict.isUnknown(sent.get(i)) ? "Unk" : "") + "Word: " + sent.get(i) + "; correct: " + correctTags[i] + "; guessed: " + finalTags[i]);
                }

                if (maxentTagger.dict.isUnknown(sent.get(i))) {
                    numWrongUnknown++;
                    if (pf != null) pf.print("*");
                }// if
            }// else
            if (pf != null) pf.print(' ');
        }// for
        if (pf != null) pf.println();

        if (verboseResults) {
            PrintWriter pw;
            try {
                pw = new PrintWriter(new OutputStreamWriter(System.out, encoding), true);
            } catch (UnsupportedEncodingException uee) {
                pw = new PrintWriter(new OutputStreamWriter(System.out), true);
            }
            pw.println(sw);
        }
    }

    /**
     * Update a confusion matrix with the errors from this sentence.
     *
     * @param confusionMatrix Confusion matrix to write to
     */
    void updateConfusionMatrix(ConfusionMatrix<String> confusionMatrix) {
        for (int i = 0; i < correctTags.length; i++)
            confusionMatrix.add(finalTags[i], correctTags[i]);
    }


    /**
     * Test using (exact Viterbi) TagInference.
     *
     * @return The tagged sentence
     */
    private ArrayList<TaggedWord> testTagInference() {
        runTagInference();
        return getTaggedSentence();
    }

    private void runTagInference() {
        initializeScorer();
        BestSequenceFinder ti = new ExactBestSequenceFinder();
        finalTags = Arrays.stream(ti.bestSequence(this)).boxed()
                .map(i -> maxentTagger.tags.getTag(i)).toArray(String[]::new);
    }

    // This is used for Dan's tag inference methods.
    // current is the actual word number + leftW
    private void setHistory(int current, History h, int[] tags) {
        //writes over the tags in the last thing in pairs
        int left = leftWindow();
        int right = rightWindow();

        for (int j = current - left; j <= current + right; j++) {
            if (j < left) {
                continue;
            } //but shouldn't happen
            if (j >= size + left) {
                break;
            } //but shouldn't happen
            h.setTag(j - left, maxentTagger.tags.getTag(tags[j]));
        }
    }

    // do initializations for the TagScorer interface
    protected void initializeScorer() {
        pairs.setSize(size);
        for (int i = 0; i < size; i++)
            pairs.setWord(i, sent.get(i));
    }

    // This scores the current assignment in PairsHolder at
    // current position h.current (returns normalized scores)
    private double[] getScores(History h) {
        double[] histories = getHistories(new String[]{}, h); // log score for each tag
        String[] tags = stringTagsAt(h.current - h.start + leftWindow());
        // tags is only used if we calculate approximate histories
        ArrayMath.logNormalize(histories);
        // assert Arrays.stream(histories).map(x-> Math.exp(x)).sum() == 1

        // now we pick out the single values for the specific tags.
        return Stream.of(tags).map(tag -> histories[maxentTagger.tags.getIndex(tag)])
                .mapToDouble(d -> d).toArray();
    }

    /**
     * This computes scores of tags at a position in a sentence (the so called "History").
     *
     * @param tags
     * @param h
     * @return
     */
    private double[] getHistories(String[] tags, History h) {
        boolean rare = maxentTagger.isRare(ExtractorFrames.cWord.extract(h));
        Extractors ex = maxentTagger.extractors;
        Extractors exR = maxentTagger.extractorsRare;
        String w = pairs.getWord(h.current);

        double[] lS = localScores.get(w);
        if (lS == null) {
            lS = getHistories(h, ex.local, rare ? exR.local : null);
            localScores.put(w, lS);
        } else if (lS.length != tags.length) {
            // This case can occur when a word was given a specific forced
            // tag, and then later it shows up without the forced tag.
            // TODO: if a word is given a forced tag, we should always get
            // its features rather than use the cache, just in case the tag
            // given is not the same tag as before
            lS = getHistories(h, ex.local, rare ? exR.local : null);
            if (tags.length > 1) {
                localScores.put(w, lS);
            }
        }
        double[] lcS = localContextScores[h.current];
        if (lcS == null) {
            lcS = getHistories(h, ex.localContext, rare ? exR.localContext : null);
            localContextScores[h.current] = lcS;
            ArrayMath.pairwiseAddInPlace(lcS, lS);
        }
        double[] totalS = getHistories(h, ex.dynamic, rare ? exR.dynamic : null);
        ArrayMath.pairwiseAddInPlace(totalS, lcS);
        return totalS;
    }

    /**
     * @param h
     * @param extractors
     * @param extractorsRare
     * @return
     */
    private double[] getHistories(History h, List<Pair<Integer, Extractor>> extractors, List<Pair<Integer, Extractor>> extractorsRare) {
        double[] scores = new double[maxentTagger.ySize];
        for (Pair<Integer, Extractor> e : extractors) {
            addScoresForExtractor(h, scores, 0, e.first(), e.second());
        }
        if (extractorsRare != null) {
            int szCommon = maxentTagger.extractors.size();  // needs to be full size list of extractors not subset of some type
            for (Pair<Integer, Extractor> e : extractorsRare) {
                addScoresForExtractor(h, scores, szCommon, e.first(), e.second());
            }
        }
        return scores;
    }

    private void addScoresForExtractor(History h, double[] scores, int szCommon, int kf, Extractor ex) {
        String val = ex.extract(h);
        int[] fAssociations = maxentTagger.fAssociations.get(kf + szCommon).get(val);

        if (fAssociations != null) {
            for (int j = 0; j < maxentTagger.ySize; j++) {
                int fNum = fAssociations[j];
                if (fNum > -1) {
                    double score = maxentTagger.getLambdaSolve().lambda[fNum];
                    scores[j] += score;
                }
            }
        }
    }


    /*
     * Implementation of the TagScorer interface follows
     */

    @Override
    public int length() {
        return sent.size();
    }

    @Override
    public int leftWindow() {
        return maxentTagger.leftContext; //hard-code for now
    }

    @Override
    public int rightWindow() {
        return maxentTagger.rightContext; //hard code for now
    }


    @Override
    public int[] getPossibleValues(int pos) {
        return Stream.of(stringTagsAt(pos))
                .map(tag -> maxentTagger.tags.getIndex(tag))
                .mapToInt(x -> x).toArray();
    }

    @Override
    public double scoreOf(int[] tags, int pos) {
        double[] scores = scoresOf(tags, pos);
        double score = Double.NEGATIVE_INFINITY;
        int[] pv = getPossibleValues(pos);
        for (int i = 0; i < scores.length; i++) {
            if (pv[i] == tags[pos]) {
                score = scores[i];
            }
        }
        return score;
    }

    @Override
    public double scoreOf(int[] sequence) {
        throw new UnsupportedOperationException();
    }

    @Override
    public double[] scoresOf(int[] tags, int pos) {
        // updating the history variable
        history.updatePointers(0, size - 1, pos - leftWindow());
        setHistory(pos, history, tags);

        // calculating scores with respect to the history
        return getScores(history);
    }

    // todo [cdm 2013]: Tagging could be sped up quite a bit here if we cached int arrays of tags by index, not Strings
    private String[] stringTagsAt(int pos) {
        pos -= leftWindow();
        // if word in padding part, return NA tag array
        if (!(0 <= pos && pos < size)) {
            return naTagArr;
        }

        // reuse supplied tags. this means each word contains only one tag, which is the supplied one.
        if (originalTags != null && originalTags.get(pos) != null) {
            return new String[]{originalTags.get(pos - leftWindow())};
        }

        String[] arr1;
        String word = sent.get(pos);
        if (maxentTagger.dict.isUnknown(word)) {
            // if word is unknown we assume all open tags
            // todo: really want array of String or int here
            arr1 = maxentTagger.tags.getOpenTags().toArray(StringUtils.EMPTY_STRING_ARRAY);
        } else {
            // if the word is known we assume it can only take tags that we seen it with during training
            arr1 = maxentTagger.dict.getTags(word);
        }
        // we expand the tags
        return maxentTagger.tags.deterministicallyExpandTags(arr1);
    }
}