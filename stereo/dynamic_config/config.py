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

def config_2_str(d, indentLevel=0, indent='    '):
    '''Convert a configuration dict into string. 
    Arguments: 
    d (dict): The dict to be printed. 
    indentLevel (int): The indent level. 0 is the first level with no indent.
    indent (str): The indent characters. 

    Returns:
    String representation.
    '''
    # Check on entry.
    assert( isinstance(d, dict) ), f'd must be a dict object. '
    assert( isinstance(indentLevel, int) ), f'indentLevel must be a positive integer.'
    assert( indentLevel >= 0 ), f'indentLevel = {indentLevel}'

    # The indent for the current level.
    localIndent = indent * indentLevel

    # Print.
    s = ''
    for key, value in d.items():
        s += localIndent
        s += f'\"{key}\": '
        if ( isinstance(value, dict) ):
            s += '{\n'
            # Recursively call itself.
            s += config_2_str(value, indentLevel+1, indent)
            s += '%s},\n' % (localIndent)
        elif ( isinstance(value, (list, tuple)) ):
            s += '['
            for v in value:
                if ( isinstance(v, dict) ):
                    s += '{\n'
                    s += config_2_str(v, indentLevel+2, indent)
                    s += '%s%s},' % (localIndent, indent)
                else:
                    s += f'{v}, '
            s += '],\n'
        else:
            s += f'{value},\n'

    return s