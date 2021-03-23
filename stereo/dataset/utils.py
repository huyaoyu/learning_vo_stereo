# coding=utf-8

# Author: Yaoyu Hu <yaoyuh@andrew.cmu.edu>
# Date: 2021-03-22

import os

def extract_strings(line, expected, delimiter):
    ss = line.split(delimiter)

    n = len(ss)

    assert (n == expected ), "{} strings extracted from {} with delimiter {}. Expected to be {}. ".format(n, line, delimiter, expected)

    result = []
    for s in ss:
        result.append( s.strip() )

    return result

def read_string_list_2D(fn, expCols, delimiter=",", prefix=""):
    """
    fn is the filename.
    expCols is the expected columns of each line. 
    delimiter is the separator between strings.
    If prefix is not empty, then the prefix string will be added to the front of each string.
    """
    
    assert (int(expCols) > 0), "expCols = {}. ".format(expCols)
    expCols = int(expCols)

    if ( False == os.path.isfile( fn ) ):
        raise Exception("%s does not exist." % (fn))
    
    strings2D = []
    n = 0

    with open(fn, "r") as fp:
        lines = fp.read().splitlines()

        n = len(lines)

        if ( "" == prefix ):
            for i in range(n):
                line = extract_strings(lines[i].strip(), expCols, delimiter)
                strings2D.append( line )
        else:
            for i in range(n):
                line = extract_strings(lines[i].strip(), expCols, delimiter)

                for j in range(expCols):
                    line[j] = "%s/%s" % ( prefix, line[j] )

                strings2D.append( line )

    if ( n == 0 ):
        raise Exception("Read {} failed. ".format(fn))

    stringCols = []
    for i in range(expCols):
        col = []
        for j in range(n):
            col.append( strings2D[j][i] )

        stringCols.append(col)

    return stringCols