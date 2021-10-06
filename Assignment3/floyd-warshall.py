# Spencer Bertsch
# Dartmouth College
# ENGS 104, Fall 2021
# floyd_warshall's implementation for directed graph

import math

i = math.inf
a = [[0, 2, 4, i],
     [1, 0, i, 1],
     [2, 5, 0, 1],
     [i, i, 3, 0]]

p = [[1, 2, 3, 4],
     [1, 2, 3, 4],
     [1, 2, 3, 4],
     [1, 2, 3, 4]]

s = 20*'-'


def pretty_print(mat, name):
    """
    Simple function to print the updated matrices after Floyd-Warshall has been run
    :param mat: list[list]] - matrix of ints
    :param name: str - name of matrix
    :return: NA
    """
    print(f'\n\n\n{name}')
    for row in mat:
        print(row)


def floyd_warshall(a, p):
    """
    Floyd Warshall method

    :param a: list[list]] - A matrix
    :param p: list[list]] - P matrix
    :return:
    """
    for j in range(len(a[0])):
        print(f'\n\n{s}  j={j+1}  {s}')
        for u in (1, 2, 3):
            for v in (1, 2, 3):
                if a[u][v] > (a[u][j] + a[j][v]):
                    print(f'{a[u][v]} is greater than {(a[u][j] + a[j][v])}! '
                          f'Updating {a[u][v]} to {(a[u][j] + a[j][v])}!')
                    # do something
                    a[u][v] = (a[u][j] + a[j][v])
                    p[u][v] = p[u][j]

                else:
                    print(f'...{a[u][v]} is less than than {(a[u][j] + a[j][v])}...')

    return a, p


if __name__ == "__main__":
    a, p = floyd_warshall(a=a, p=p)
    pretty_print(mat=a, name='NEW MATRIX A')
    pretty_print(mat=p, name='NEW MATRIX P')

