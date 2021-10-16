

def test_graphic_vector(vec: list) -> bool:
    """
    Return True if the input vector is a graphic vector, False otherwise

    :param: vec - list of ints representing the vector to test
    :return: bool - True if the input vector is a graphic vector, False otherwise
    """

    # sort the vector in case the initial vector is unsorted
    vec.sort(reverse=True)

    print(f'INPUT VECTOR: {vec}')
    # test for negatives
    v: list = [x for x in vec if x<0]
    if len(v) > 0:
        print(f'{vec} is NOT a graphic vector.')
        return False

    z: list = [x for x in vec if x==0]
    if len(z)==len(vec):
        print(f'{vec} IS a graphic vector!!!')
        return True

        # pop the first item in the list
    a = vec.pop(0)

    # decrement the first (a) elements of the list where (a)
    # is the number we just popped
    new_vec = []
    for i, element in enumerate(vec):
        if (i) < a:
            element = element-1
        new_vec.append(element)

    # sort the vector
    new_vec.sort(reverse=True)
    print(vec, '\n')

    # make the recursive call
    test_graphic_vector(vec=new_vec)


if __name__ == "__main__":
    test_graphic_vector(vec=[5, 4, 4, 3, 3, 3, 2])
