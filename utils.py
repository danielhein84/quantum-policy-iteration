'''
(C) Copyright Siemens AG 2023

SPDX-License-Identifier: MIT
'''
def bin_float_to_dec_float(bin_float):
    """
    Convert binary float to a decimal float
    :param bin_float: binary float as a string
    :return: decimal float as a float
    """
    bin_whole, bin_frac = bin_float.split(".")
    dec_whole = int(bin_whole, 2)
    dec_frac = int(bin_frac, 2) / 2 ** len(bin_frac)
    dec_float = dec_whole + dec_frac
    return dec_float


def verbose_print(txt, verbose):
    if verbose:
        print(txt)
