package edu.stanford.nlp.tagger.maxent

import edu.stanford.nlp.util.logging.Redwood

import edu.stanford.nlp.maxent.Feature
import edu.stanford.nlp.maxent.Problem
import edu.stanford.nlp.maxent.iis.LambdaSolve

import java.text.NumberFormat
import java.io.DataInputStream


/**
 * This module does the working out of lambda parameters for binary tagger
 * features.  It can use either IIS or CG.
 *
 * @author Kristina Toutanova
 * @version 1.0
 */
class LambdaSolveTagger : LambdaSolve {

    /**
     * Suppress extraneous printouts
     */
    //@SuppressWarnings("unused")
    //private static final boolean VERBOSE = false;


    internal constructor(p1: Problem, eps1: Double, fnumArr: Array<ByteArray>) {
        p = p1
        eps = eps1
        // newtonerr = nerr1;
        lambda = DoubleArray(p1.fSize)
        // lambda_converged = new boolean[p1.fSize];
        // cdm 2008: Below line is memory hog. Is there anything we can do to avoid this square array allocation?
        probConds = Array(p1.data.xSize) { DoubleArray(p1.data.ySize) }
        this.fnumArr = fnumArr
        zlambda = DoubleArray(p1.data.xSize)
        ftildeArr = DoubleArray(p.fSize)
        initCondsZlambdaEtc()
        super.setBinary()
    }


    /** Initialize a trained LambdaSolveTagger.
     * This is the version used when loading a saved tagger.
     * Only the lambda array is used, and the rest is irrelevant, CDM thinks.
     *
     * @param dataStream Stream to load lambda parameters from.
     */
    internal constructor(dataStream: DataInputStream) {
        lambda = LambdaSolve.read_lambdas(dataStream)
        super.setBinary()
    }

    /** Initialize a trained LambdaSolveTagger.
     * This is the version used when creating a LambdaSolveTagger from
     * a condensed lambda array.
     * Only the lambda array is used, and the rest is irrelevant, CDM thinks.
     *
     * @param lambda Array used as the lambda parameters (directly; no safety copy is made).
     */
    internal constructor(lambda: DoubleArray) {
        this.lambda = lambda
        super.setBinary()
    }

    private fun initCondsZlambdaEtc() {
        // updatePointers pcond
        for (x in 0 until p.data.xSize) {
            for (y in 0 until p.data.ySize) {
                probConds[x][y] = 1.0 / p.data.ySize
            }
        }
        log.info(" pcond initialized ")
        // updatePointers zlambda
        for (x in 0 until p.data.xSize) {
            zlambda[x] = p.data.ySize.toDouble()
        }
        log.info(" zlambda initialized ")
        // updatePointers ftildeArr
        for (i in 0 until p.fSize) {
            ftildeArr[i] = p.functions.get(i).ftilde()
            if (ftildeArr[i] == 0.0) {
                log.info(" Empirical expectation 0 for feature $i")
            }
        }
        log.info(" ftildeArr initialized ")
    }

    internal fun g(lambdaP: Double, index: Int): Double {
        var s = 0.0
        for (i in 0 until p.functions.get(index).len()) {
            val y = (p.functions.get(index) as TaggerFeature).yTag
            val x = p.functions.get(index).getX(i)
            s = s + p.data.ptildeX(x) * pcond(y, x) * 1.0 * Math.exp(lambdaP * fnum(x, y))
        }
        s = s - ftildeArr[index]

        return s
    }

    internal fun fExpected(f: Feature): Double {
        val tF = f as TaggerFeature
        var s = 0.0
        val y = tF.yTag
        for (i in 0 until f.len()) {
            val x = tF.getX(i)
            s = s + p.data.ptildeX(x) * pcond(y, x)
        }
        return s
    }


    /** Works out whether the model expectations match the empirical
     * expectations.
     * @return Whether the model is correct
     */
    override fun checkCorrectness(): Boolean {
        log.info("Checking model correctness; x size " + p.data.xSize + ' '.toString() + ", ysize " + p.data.ySize)

        val nf = NumberFormat.getNumberInstance()
        nf.maximumFractionDigits = 4
        var flag = true
        for (f in lambda.indices) {
            if (Math.abs(lambda[f]) > 100) {
                log.info(" Lambda too big " + lambda[f])
                log.info(" empirical " + ftildeArr[f] + " expected " + fExpected(p.functions.get(f)))
            }
        }

        for (i in ftildeArr.indices) {
            val exp = Math.abs(ftildeArr[i] - fExpected(p.functions.get(i)))
            if (exp > 0.001) {
                flag = false
                log.info("Constraint " + i + " not satisfied emp " + nf.format(ftildeArr[i]) + " exp " + nf.format(fExpected(p.functions.get(i))) + " diff " + nf.format(exp) + " lambda " + nf.format(lambda[i]))
            }
        }
        for (x in 0 until p.data.xSize) {
            var s = 0.0
            for (y in 0 until p.data.ySize) {
                s = s + probConds[x][y]
            }
            if (Math.abs(s - 1) > 0.0001) {
                for (y in 0 until p.data.ySize) {
                    log.info("$y : " + probConds[x][y])
                }
                log.info("probabilities do not sum to one " + x + ' '.toString() + s.toFloat())
            }
        }
        return flag
    }

    companion object {
        /** A logger for this class  */
        private val log = Redwood.channels(LambdaSolveTagger::class.java)
    }
}

