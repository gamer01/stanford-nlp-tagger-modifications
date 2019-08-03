/*
 * Title:       StanfordMaxEnt<p>
 * Description: A Maximum Entropy Toolkit<p>
 * Copyright:   The Board of Trustees of The Leland Stanford Junior University
 * Company:     Stanford University<p>
 */

package edu.stanford.nlp.tagger.maxent;

import edu.stanford.nlp.ling.WordTag;

import java.util.*;

/**
 * A simple class that maintains a list of WordTag pairs which are interned
 * as they are added.  This stores a tagged corpus.
 * It is also used to represent partial histories at tagging time.
 * It may not simply represent a sentence, and a History is used to overlay a sentence on a PairsHolder.
 *
 * @author Kristina Toutanova
 * @version 1.0
 */
public class PairsHolder {

    // todo: In Java 5+, just make this class an ArrayList<WordTag> and be done with it?? Or actually, probably a PaddedList. Or need a WindowedList?

    private final ArrayList<WordTag> arr = new ArrayList<>();

    public PairsHolder() {
    }

    // todo: This method seems crazy.  Can't we either just do nothing or using ensureCapacity()?
    public void setSize(int s) {
        while (arr.size() < s) {
            arr.add(new WordTag(null, "NN"));  // todo: remove NN.  NA okay?
        }
    }

    public int size() {
        return arr.size();
    }

    void clear() {
        arr.clear();
    }


    void add(WordTag wordtag) {
        arr.add(wordtag);
    }

    void setWord(int pos, String word) {
        arr.get(pos).setWord(word);
    }

    void setTag(int pos, String tag) {
        arr.get(pos).setTag(tag);
    }

    String getTag(int position) {
        return arr.get(position).tag();
    }

    /**
     * This gets a word at an absolute position.
     */
    String getWord(int position) {
        return arr.get(position).word();
    }

    /**
     * This gets a word at a position relative to the "current" position in the history.
     */
    String getWord(History h, int position) {
        final int p = h.current + position;
        return (p >= h.start && p <= h.end) ? arr.get(p).word() : "NA";
    }

    String getTag(History h, int position) {
        final int p = h.current + position;
        return (p >= h.start && p <= h.end) ? arr.get(p).tag() : "NA";
    }

    @Override
    public String toString() {
        return arr.toString();
    }
}
