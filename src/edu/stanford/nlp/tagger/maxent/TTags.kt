package edu.stanford.nlp.tagger.maxent

import edu.stanford.nlp.io.IOUtils
import edu.stanford.nlp.io.RuntimeIOException
import edu.stanford.nlp.tagger.common.Tagger
import edu.stanford.nlp.util.Generics
import edu.stanford.nlp.util.HashIndex
import edu.stanford.nlp.util.Index
import java.io.*

import java.util.*
import kotlin.streams.toList

class TTags// conjunctions
/**
// conjunctions
 * This class holds the POS tags, assigns them unique ids, and knows which tags
 * are open versus closed class.
 *
 *
 * Title:        StanfordMaxEnt
 *
 *
 * Description:  A Maximum Entropy Toolkit
 *
 *
 * Company:      Stanford University
 *
 *
 *
 * @author Kristina Toutanova
 * @version 1.0
 */

// punctuation
// Using the french treebank, with Spence's adaptations of
// Candito's treebank modifications, we get that only the
// punctuation tags are reliably closed:
// !, ", *, ,, -, -LRB-, -RRB-, ., ..., /, :, ;, =, ?, [, ]
// The current version of the German tagger is built with the
// negra-tiger data set.  We use the STTS tag set.  In
// particular, we use the version with the changes described in
// appendix A-2 of
// http://www.uni-potsdam.de/u/germanistik/ls_dgs/tiger1-intro.pdf
// eg the STTS tag set with PROAV instead of PAV
// To find the closed tags, we use lists of standard closed German
// tags, eg
// http://www.sfs.uni-tuebingen.de/Elwis/stts/Wortlisten/WortFormen.html
// In other words:
//
// APPO APPR APPRART APZR ART KOKOM KON KOUI KOUS PDAT PDS PIAT
// PIDAT PIS PPER PPOSAT PPOSS PRELAT PRELS PRF PROAV PTKA
// PTKANT PTKNEG PTKVZ PTKZU PWAT PWAV PWS VAFIN VAIMP VAINF
// VAPP VMFIN VMINF VMPP
//
// One issue with this is that our training data does not have
// the complete collection of many of these closed tags.  For
// example, words with the tag APPR show up in the test or dev
// sets without ever showing up in the training.  Tags that
// don't have this property:
//
// KOKOM PPOSS PTKA PTKNEG PWAT VAINF VAPP VMINF VMPP
// this shouldn't be a tag of the dataset, but was a conversion bug!
// kulick tag set
// the following tags seem to be complete sets in the training
// data (see the comments for "german" for more info)
// maybe more should still be added ... cdm jun 2006
/* chinese treebank 5 tags *///  closed.add("IN");

// punctuation
// Using the french treebank, with Spence's adaptations of
// Candito's treebank modifications, we get that only the
// punctuation tags are reliably closed:
// !, ", *, ,, -, -LRB-, -RRB-, ., ..., /, :, ;, =, ?, [, ]
// The current version of the German tagger is built with the
// negra-tiger data set.  We use the STTS tag set.  In
// particular, we use the version with the changes described in
// appendix A-2 of
// http://www.uni-potsdam.de/u/germanistik/ls_dgs/tiger1-intro.pdf
// eg the STTS tag set with PROAV instead of PAV
// To find the closed tags, we use lists of standard closed German
// tags, eg
// http://www.sfs.uni-tuebingen.de/Elwis/stts/Wortlisten/WortFormen.html
// In other words:
//
// APPO APPR APPRART APZR ART KOKOM KON KOUI KOUS PDAT PDS PIAT
// PIDAT PIS PPER PPOSAT PPOSS PRELAT PRELS PRF PROAV PTKA
// PTKANT PTKNEG PTKVZ PTKZU PWAT PWAV PWS VAFIN VAIMP VAINF
// VAPP VMFIN VMINF VMPP
//
// One issue with this is that our training data does not have
// the complete collection of many of these closed tags.  For
// example, words with the tag APPR show up in the test or dev
// sets without ever showing up in the training.  Tags that
// don't have this property:
//
// KOKOM PPOSS PTKA PTKNEG PWAT VAINF VAPP VMINF VMPP
// this shouldn't be a tag of the dataset, but was a conversion bug!
// kulick tag set
// the following tags seem to be complete sets in the training
// data (see the comments for "german" for more info)
// maybe more should still be added ... cdm jun 2006
/* chinese treebank 5 tags *///  closed.add("IN");// conjunctions

// punctuation/* add closed-class lists for other languages here */
// Using the french treebank, with Spence's adaptations of
// Candito's treebank modifications, we get that only the
// punctuation tags are reliably closed:
// !, ", *, ,, -, -LRB-, -RRB-, ., ..., /, :, ;, =, ?, [, ]
// The current version of the German tagger is built with the
// negra-tiger data set.  We use the STTS tag set.  In
// particular, we use the version with the changes described in
// appendix A-2 of
// http://www.uni-potsdam.de/u/germanistik/ls_dgs/tiger1-intro.pdf
// eg the STTS tag set with PROAV instead of PAV
// To find the closed tags, we use lists of standard closed German
// tags, eg
// http://www.sfs.uni-tuebingen.de/Elwis/stts/Wortlisten/WortFormen.html
// In other words:
//
// APPO APPR APPRART APZR ART KOKOM KON KOUI KOUS PDAT PDS PIAT
// PIDAT PIS PPER PPOSAT PPOSS PRELAT PRELS PRF PROAV PTKA
// PTKANT PTKNEG PTKVZ PTKZU PWAT PWAV PWS VAFIN VAIMP VAINF
// VAPP VMFIN VMINF VMPP
//
// One issue with this is that our training data does not have
// the complete collection of many of these closed tags.  For
// example, words with the tag APPR show up in the test or dev
// sets without ever showing up in the training.  Tags that
// don't have this property:
//
// KOKOM PPOSS PTKA PTKNEG PWAT VAINF VAPP VMINF VMPP
// this shouldn't be a tag of the dataset, but was a conversion bug!
// kulick tag set
// the following tags seem to be complete sets in the training
// data (see the comments for "german" for more info)
// maybe more should still be added ... cdm jun 2006
/* chinese treebank 5 tags *///  closed.add("IN");
@JvmOverloads constructor(config: TaggerConfig, language: String = "") {

    private var index: Index<String>
    private val closed: MutableSet<String>

    private var _openTags: MutableSet<String>? = null

    val openTags: MutableSet<String>
        get() {
            val open = Generics.newHashSet<String>()
            if (_openTags == null) {
                for (tag in index) {
                    if (!closed.contains(tag)) {
                        open.add(tag)
                    }
                }
                _openTags = open
            }
            return _openTags!!
        }
    private val isEnglish: Boolean // for speed
    /** If true, then the open tags are fixed and we set closed tags based on
     * index-openTags; otherwise, we set open tags based on index-closedTags.
     */
    private var openFixed = false


    /** When making a decision based on the training data as to whether a
     * tag is closed, this is the threshold for how many tokens can be in
     * a closed class - purposely conservative.
     * TODO: make this an option you can set; need to pass in TaggerConfig object and then can say = config.getClosedTagThreshold());
     */
    private val closedTagThreshold: Int
    private val doDeterministicTagExpansion: Boolean
    /** If true, when a model is trained, all tags that had fewer tokens than
     * closedTagThreshold will be considered closed.
     */
    private var expansionRules = listOf<Set<String>>()

    private var learnClosedTags = false

    val size: Int
        get() = index.size()


    /** Return the Set of tags used by this tagger (available after training the tagger).
     *
     * @return The Set of tags used by this tagger
     */
    fun tagSet(): Set<String> {
        return HashSet(index.objectsList())
    }

    fun add(tag: String): Int {
        return index.addToIndex(tag)
    }

    fun getTag(i: Int): String {
        return index.get(i)
    }

    fun save(filename: String,
             tagTokens: Map<String, Set<String>>) {
        try {
            val out = IOUtils.getDataOutputStream(filename)
            save(out, tagTokens)
            out.close()
        } catch (e: IOException) {
            throw RuntimeIOException(e)
        }

    }

    fun save(file: DataOutputStream,
             tagTokens: Map<String, Set<String>>) {
        try {
            file.writeInt(index.size())
            for (item in index) {
                file.writeUTF(item)
                if (learnClosedTags) {
                    tagTokens[item]?.let {
                        if (it.size < closedTagThreshold) {
                            markClosed(item)
                        }
                    }
                }
                file.writeBoolean(isClosed(item))
            }
        } catch (e: IOException) {
            throw RuntimeIOException(e)
        }

    }


    fun read(filename: String) {
        try {
            val `in` = IOUtils.getDataInputStream(filename)
            read(`in`)
            `in`.close()
        } catch (e: IOException) {
            throw RuntimeIOException(e)
        }

    }

    fun read(file: DataInputStream) {
        try {
            val size = file.readInt()
            index = HashIndex()
            for (i in 0 until size) {
                val tag = file.readUTF()
                val inClosed = file.readBoolean()
                index.add(tag)

                if (inClosed) closed.add(tag)
            }
        } catch (e: IOException) {
            throw RuntimeIOException(e)
        }

    }


    fun isClosed(tag: String): Boolean {
        return if (openFixed) {
            !openTags.contains(tag)
        } else {
            closed.contains(tag)
        }
    }

    internal fun markClosed(tag: String) {
        add(tag)
        closed.add(tag)
    }

    fun setLearnClosedTags(learn: Boolean) {
        learnClosedTags = learn
    }

    fun setOpenClassTags(openClassTags: Array<String>) {
        openTags.addAll(Arrays.asList(*openClassTags))
        for (tag in openClassTags) {
            add(tag)
        }
        openFixed = true
    }

    fun setClosedClassTags(closedClassTags: Array<String>) {
        for (tag in closedClassTags) {
            markClosed(tag)
        }
    }


    fun indexOf(tag: String): Int {
        return index.indexOf(tag)
    }

    /**
     * Deterministically adds other possible tags for words given observed tags.
     * For instance, for English with the Penn POS tag, a word with the VB
     * tag would also be expected to have the VBP tag.
     *
     *
     * The current implementation is a bit contorted, as it works to avoid
     * object allocations wherever possible for maximum runtime speed. But
     * intuitively it's just: For English (only),
     * if the VBD tag is present but not VBN, add it, and vice versa;
     * if the VB tag is present but not VBP, add it, and vice versa.
     *
     * @param tags Known possible tags for the word
     * @return A superset of tags
     */
    fun deterministicallyExpandTags(tags: Array<String>): Array<String> {
        if (doDeterministicTagExpansion) {
            val oldTags = tags.toSet()
            val newTags = oldTags.toMutableSet()

            /*oldTags.forEach {
                if (it.endsWith('*')) {
                    // e.g. ('KO*' → 'KO*', 'KOKOM', 'KON', 'KOUS')
                    val pattern = it.replace("*", ".+")
                    newTags.addAll(tags.filter { it.toRegex().matches(pattern) })
                } else {
                    // e.g. ('VVPP' → 'VVPP', 'ADJA<VVPP')
                    newTags.addAll(tags.filter { it.toRegex().matches("(.+<|)$it") })
                }
            }*/

            expansionRules.forEach {
                if (oldTags.intersect(it).isNotEmpty())
                    newTags.addAll(it)
            }

            return newTags.toTypedArray()
        }

        return tags
    }

    override fun toString(): String {
        val s = StringBuilder()
        s.append(index)
        s.append(' ')
        if (openFixed) {
            s.append(" OPEN:").append(openTags)
        } else {
            s.append(" open:").append(openTags).append(" CLOSED:").append(closed)
        }
        return s.toString()
    }

    init {
        this.index = HashIndex()
        this.closed = Generics.newHashSet<String>()
        this.closedTagThreshold = config.closedTagThreshold
        this.doDeterministicTagExpansion = config.doDeterministicTagExpansion

        if (this.doDeterministicTagExpansion) {
            try {
                this.expansionRules = File(config.tagExpansionRuleFile).bufferedReader().lines().map {
                    it.split(',').map { it.strip() }.toSet()
                }.toList()
            } catch (e: FileNotFoundException) {
                this.expansionRules = listOf()
            }
        }

        if (language.equals("english", ignoreCase = true)) {
            closed.add(".")
            closed.add(",")
            closed.add("``")
            closed.add("''")
            closed.add(":")
            closed.add("$")
            closed.add("EX")
            closed.add("(")
            closed.add(")")
            closed.add("#")
            closed.add("MD")
            closed.add("CC")
            closed.add("DT")
            closed.add("LS")
            closed.add("PDT")
            closed.add("POS")
            closed.add("PRP")
            closed.add("PRP$")
            closed.add("RP")
            closed.add("TO")
            closed.add(Tagger.EOS_TAG)
            closed.add("UH")
            closed.add("WDT")
            closed.add("WP")
            closed.add("WP$")
            closed.add("WRB")
            closed.add("-LRB-")
            closed.add("-RRB-")
            //  closed.add("IN");
            isEnglish = true
        } else if (language.equals("polish", ignoreCase = true)) {
            closed.add(".")
            closed.add(",")
            closed.add("``")
            closed.add("''")
            closed.add(":")
            closed.add("$")
            closed.add("(")
            closed.add(")")
            closed.add("#")
            closed.add("POS")
            closed.add(Tagger.EOS_TAG)
            closed.add("ppron12")
            closed.add("ppron3")
            closed.add("siebie")
            closed.add("qub")
            closed.add("conj")
            isEnglish = false
        } else if (language.equals("chinese", ignoreCase = true)) {
            /* chinese treebank 5 tags */
            closed.add("AS")
            closed.add("BA")
            closed.add("CC")
            closed.add("CS")
            closed.add("DEC")
            closed.add("DEG")
            closed.add("DER")
            closed.add("DEV")
            closed.add("DT")
            closed.add("ETC")
            closed.add("IJ")
            closed.add("LB")
            closed.add("LC")
            closed.add("P")
            closed.add("PN")
            closed.add("PU")
            closed.add("SB")
            closed.add("SP")
            closed.add("VC")
            closed.add("VE")
            isEnglish = false
        } else if (language.equals("arabic", ignoreCase = true)) {
            // kulick tag set
            // the following tags seem to be complete sets in the training
            // data (see the comments for "german" for more info)
            closed.add("PUNC")
            closed.add("CC")
            closed.add("CPRP$")
            closed.add(Tagger.EOS_TAG)
            // maybe more should still be added ... cdm jun 2006
            isEnglish = false
        } else if (language.equals("german", ignoreCase = true)) {
            // The current version of the German tagger is built with the
            // negra-tiger data set.  We use the STTS tag set.  In
            // particular, we use the version with the changes described in
            // appendix A-2 of
            // http://www.uni-potsdam.de/u/germanistik/ls_dgs/tiger1-intro.pdf
            // eg the STTS tag set with PROAV instead of PAV
            // To find the closed tags, we use lists of standard closed German
            // tags, eg
            // http://www.sfs.uni-tuebingen.de/Elwis/stts/Wortlisten/WortFormen.html
            // In other words:
            //
            // APPO APPR APPRART APZR ART KOKOM KON KOUI KOUS PDAT PDS PIAT
            // PIDAT PIS PPER PPOSAT PPOSS PRELAT PRELS PRF PROAV PTKA
            // PTKANT PTKNEG PTKVZ PTKZU PWAT PWAV PWS VAFIN VAIMP VAINF
            // VAPP VMFIN VMINF VMPP
            //
            // One issue with this is that our training data does not have
            // the complete collection of many of these closed tags.  For
            // example, words with the tag APPR show up in the test or dev
            // sets without ever showing up in the training.  Tags that
            // don't have this property:
            //
            // KOKOM PPOSS PTKA PTKNEG PWAT VAINF VAPP VMINF VMPP
            closed.add("$,")
            closed.add("$.")
            closed.add("$(")
            closed.add("--") // this shouldn't be a tag of the dataset, but was a conversion bug!
            closed.add(Tagger.EOS_TAG)
            closed.add("KOKOM")
            closed.add("PPOSS")
            closed.add("PTKA")
            closed.add("PTKNEG")
            closed.add("PWAT")
            closed.add("VAINF")
            closed.add("VAPP")
            closed.add("VMINF")
            closed.add("VMPP")
            isEnglish = false
        } else if (language.equals("french", ignoreCase = true)) {
            // Using the french treebank, with Spence's adaptations of
            // Candito's treebank modifications, we get that only the
            // punctuation tags are reliably closed:
            // !, ", *, ,, -, -LRB-, -RRB-, ., ..., /, :, ;, =, ?, [, ]
            closed.add("!")
            closed.add("\"")
            closed.add("*")
            closed.add(",")
            closed.add("-")
            closed.add("-LRB-")
            closed.add("-RRB-")
            closed.add(".")
            closed.add("...")
            closed.add("/")
            closed.add(":")
            closed.add(";")
            closed.add("=")
            closed.add("?")
            closed.add("[")
            closed.add("]")
            isEnglish = false
        } else if (language.equals("spanish", ignoreCase = true)) {
            closed.add(Tagger.EOS_TAG)

            // conjunctions
            closed.add("cc")
            closed.add("cs")

            // punctuation
            closed.add("faa")
            closed.add("fat")
            closed.add("fc")
            closed.add("fca")
            closed.add("fct")
            closed.add("fd")
            closed.add("fe")
            closed.add("fg")
            closed.add("fh")
            closed.add("fia")
            closed.add("fit")
            closed.add("fla")
            closed.add("flt")
            closed.add("fp")
            closed.add("fpa")
            closed.add("fpt")
            closed.add("fra")
            closed.add("frc")
            closed.add("fs")
            closed.add("ft")
            closed.add("fx")
            closed.add("fz")

            isEnglish = false
        } else if (language.equals("medpost", ignoreCase = true)) {
            closed.add(".")
            closed.add(",")
            closed.add("``")
            closed.add("''")
            closed.add(":")
            closed.add("$")
            closed.add("EX")
            closed.add("(")
            closed.add(")")
            closed.add("VM")
            closed.add("CC")
            closed.add("DD")
            closed.add("DB")
            closed.add("GE")
            closed.add("PND")
            closed.add("PNG")
            closed.add("TO")
            closed.add(Tagger.EOS_TAG)
            closed.add("-LRB-")
            closed.add("-RRB-")
            isEnglish = false
        } else if (language.equals("testing", ignoreCase = true)) {
            closed.add(".")
            closed.add(Tagger.EOS_TAG)
            isEnglish = false
        } else if (language.equals("", ignoreCase = true)) {
            isEnglish = false
        } else {
            throw RuntimeException("unknown language: $language")
        }
    }

}
