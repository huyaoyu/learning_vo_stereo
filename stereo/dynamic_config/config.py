# coding=utf-8

# Author: Yaoyu Hu <yaoyuh@andrew.cmu.edu>
# Date: 2021-03-20

def show_config(d, indentLevel=0, indent='    '):
    '''Print a configuration dict. 
    Arguments: 
    d (dict): The dict to be printed. 
    indentLevel (int): The indent level. 0 is the first level with no indent.
    indent (str): The indent characters. 

    Returns:
    None
    '''
    # Check on entry.
    assert( isinstance(d, dict) ), f'd must be a dict object. '
    assert( isinstance(indentLevel, int) ), f'indentLevel must be a positive integer.'
    assert( indentLevel >= 0 ), f'indentLevel = {indentLevel}'

    # The indent for the current level.
    localIndent = indent * indentLevel

    # Print.
    for key, value in d.items():
        print(localIndent, end='')
        print(f'\"{key}\": ', end='')
        if ( isinstance(value, dict) ):
            print('{')
            # Recursively call itself.
            show_config(value, indentLevel+1, indent)
            print('},')
            continue

        print(f'{value},')
