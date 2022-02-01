"""One-off utilities."""

from itertools import takewhile

def find_common_prefix(list_of_strings: list):
    # https://www.geeksforgeeks.org/python-ways-to-determine-common-prefix-in-set-of-strings/
    res = ''.join(c[0] for c in takewhile(lambda x:
        all(x[0] == y for y in x), zip(*list_of_strings)))
    return res
