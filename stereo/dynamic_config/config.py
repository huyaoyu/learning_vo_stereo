# coding=utf-8

# Author: Yaoyu Hu <yaoyuh@andrew.cmu.edu>
# Date: 2021-03-20

def show_config(d, indentLevel=0, indent='    '):
    assert( isinstance(d, dict) ), f'd must be a dict object. '
    assert( isinstance(indentLevel, int) ), f'indentLevel must be a positive integer.'
    assert( indentLevel >= 0 ), f'indentLevel = {indentLevel}'

    localIndent = indent * indentLevel

    for key, value in d.items():
        print(localIndent, end='')
        print(f'\"{key}\": ', end='')
        if ( isinstance(value, dict) ):
            print('{')
            show_config(value, indentLevel+1, indent)
            print('},')
            continue

        print(f'{value},')
