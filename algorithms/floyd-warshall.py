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

p = [[1, 2, 3, 0],
     [1, 2, 0, 4],
     [1, 2, 3, 4],
     [0, 0, 3, 4]]

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
    V = range(len(a[0]))
    for j in V:
        print(f'\n\n{s}  j={j+1}  {s}')
        v_copy = list(V).copy()
        v_copy.remove(j)

        for u in v_copy:
            for v in v_copy:
                if a[u][v] > (a[u][j] + a[j][v]):
                    print(f'{a[u][v]} is greater than {(a[u][j] + a[j][v])}! '
                          f'Updating {a[u][v]} to {(a[u][j] + a[j][v])}!')
                    # update the a matrix
                    a[u][v] = (a[u][j] + a[j][v])
                    print(f'Updating P matrix: {p[u][v]} to {(p[u][j])}!')
                    # update the p matrix
                    p[u][v] = p[u][j]
                else:
                    print(f'...{a[u][v]} is less than than {(a[u][j] + a[j][v])}...')

    return a, p


if __name__ == "__main__":

    pretty_print(mat=a, name='ORIGINAL MATRIX A')
    pretty_print(mat=p, name='ORIGINAL MATRIX P')
    a, p = floyd_warshall(a=a, p=p)
    pretty_print(mat=a, name='NEW MATRIX A')
    pretty_print(mat=p, name='NEW MATRIX P')
