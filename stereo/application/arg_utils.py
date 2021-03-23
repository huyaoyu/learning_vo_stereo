# coding=utf-8

# Author: Yaoyu Hu <yaoyuh@andrew.cmu.edu>
# Date: 2021-03-23

import os

def convert_str_2_int_list(s, d=","):
    """
    Convert a string of integers into a list.
    s: The input string.
    d: The delimiter.
    """

    ss = s.split(d)

    temp = []

    for t in ss:
        temp.append( int(t) )

    return temp

def read_string_list(fn, prefix=""):
    """
    Read a file contains lines of strings. A list will be returned.
    Each element of the list contains a single entry of the input file.
    Leading and tailing white spaces, tailing carriage return will be stripped.
    """

    if ( False == os.path.isfile( fn ) ):
        raise Exception("%s does not exist." % (fn))
    
    with open(fn, "r") as fp:
        lines = fp.read().splitlines()

        n = len(lines)

        if ( "" == prefix ):
            for i in range(n):
                lines[i] = lines[i].strip()
        else:
            for i in range(n):
                lines[i] = "%s/%s" % ( prefix, lines[i].strip() )

    return lines

def extract_integers_from_argument(arg, expected=0):
    """
    arg is a string separated by commas.
    expected is the expected number to be extracted. Set 0 to disable.
    """

    ss = arg.split(",")

    ns = [ int(s.strip()) for s in ss ]

    if ( expected > 0 and len(ns) != expected ):
        raise Exception("Expecting {} integers. {} extracted from {}. ".format(expected, len(ns), arg))

    return ns

def valid_override(arg):
    return ( arg == 0 or arg == 1 )