//AmbiguityClasses -- StanfordMaxEnt, A Maximum Entropy Toolkit
//Copyright (c) 2002-2008 Leland Stanford Junior University


//This program is free software; you can redistribute it and/or
//modify it under the terms of the GNU General Public License
//as published by the Free Software Foundation; either version 2
//of the License, or (at your option) any later version.

//This program is distributed in the hope that it will be useful,
//but WITHOUT ANY WARRANTY; without even the implied warranty of
//MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
//GNU General Public License for more details.

//You should have received a copy of the GNU General Public License
//along with this program; if not, write to the Free Software
//Foundation, Inc., 59 Temple Place - Suite 330, Boston, MA  02111-1307, USA.

//For more information, bug reports, fixes, contact:
//Christopher Manning
//Dept of Computer Science, Gates 1A
//Stanford CA 94305-9010
//USA
//Support/Questions: java-nlp-user@lists.stanford.edu
//Licensing: java-nlp-support@lists.stanford.edu
//http://www-nlp.stanford.edu/software/tagger.shtml


package edu.stanford.nlp.tagger.maxent;

import edu.stanford.nlp.util.Index;
import edu.stanford.nlp.util.HashIndex;

/**
 * A collection of Ambiguity Class.
 * <i>The code currently here is rotted and would need to be revived.</i>
 *
 * @author Kristina Toutanova
 * @version 1.0
 */

// TODO: if it's rotted and not used anywhere, can we just get rid of it all?  [CDM: It would be nice to keep and revive someday. It is a nice and sometimes useful idea.]

public class AmbiguityClasses {

  private final Index<AmbiguityClass> classes;
  private static final String naWord = Defaults.naTag;

  // TODO: this isn't used anywhere, either
  // protected final AmbiguityClass naClass = new AmbiguityClass(null, false, null, null);

  public AmbiguityClasses(TTags ttags) {
    classes = new HashIndex<>();
    // naClass.updatePointers(naWord, ttags);
  }

  private int add(AmbiguityClass a) {
    if(classes.contains(a)) {
      return classes.indexOf(a);
    }
    classes.add(a);
    return classes.indexOf(a);
  }

  protected int getClass(String word, Dictionary dict, int veryCommonWordThresh, TTags ttags) {
    if (word.equals(naWord)) {
      return -2;
    }
    if (dict.isUnknown(word)) {
      return -1;
    }
    boolean veryCommon = dict.sum(word) > veryCommonWordThresh;
    AmbiguityClass a = new AmbiguityClass(word, veryCommon, dict, ttags);
    // TODO: surely it would be faster and not too expensive to cache
    // the results of creating a whole bunch of these, since we're
    // probably constructing the same AmbiguityClass multiple times
    // for each word.  Furthermore, the separation of having two
    // constructors here is pretty awful, quite frankly.
    return add(a);
  }

}
