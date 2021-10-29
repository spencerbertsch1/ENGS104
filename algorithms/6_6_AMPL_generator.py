# Script to produce AMPL code programmatically
# Spencer Bertsch
# ENGS 104
# Fall 2021 @ Dartmouth College


def write_6_6(coefficients: list, group_array: list, num_students: int):

    f = open("6_6_AMPL_code.mod", "a")

    f.write(f'# define variables here \n')
    for i in range(len(coefficients)):
        f.write(f'var x{i+1} binary; \n')

    f.write(f'\n')
    f.write(f'# now we want to maximize the score for each student \n')
    f.write(f'maximize student_scores: \n')
    for i, coefficient in enumerate(coefficients):
        if i < len(coefficients) - 1:
            f.write(f'{coefficient}*x{i+1} + ')
        else:
            f.write(f'{coefficient}*x{i+1};')

    f.write(f'\n\n')
    f.write(f'# we can now add the constraints - one for each student. \n')
    group_no = 1
    for student in range(num_students):
        student += 1
        f.write(f'subject to Student{student}: ')
        for j in range(4):
            if j < 3:
                f.write(f'x{group_no} + ')
            else:
                f.write(f'x{group_no} = 1; \n')
            group_no += 1

    f.write(f'\n\n')
    f.write(f'# and finally we can add the lab constraints - one for each lab group. \n')
    group_no = 1
    for group in range(4):
        group += 1
        f.write(f'subject to G{group}: ')
        for j in range(len(coefficients)):
            if j < 17:
                f.write(f'x{j*4 + group} + ')
            elif j == 17:
                f.write(f'x{j*4 + group} <= {group_array[group-1]}; \n')
            else:
                pass
            group_no += 1

    f.write(f'')
    f.close()


if __name__ == "__main__":
    # define student preferences
    coefficients: list = [5,8,2,10,10,7,5,1,3,10,8,8,10,6,2,10,2,10,9,4,6,10,6,6,10,7,1,5,4,6,8,10,7,10,3,10,1,2,4,10,6,
                          9,10,3,4,7,2,10,10,8,3,7,10,9,8,7,7,10,4,10,10,5,5,1,10,2,6,4,10,3,7,3]

    # define groups
    group_array: list = [4,4,7,4]

    # define number of students
    num_students: int = 18

    # create the AMPL code and write it to a .mod file
    write_6_6(coefficients=coefficients, group_array=group_array, num_students=num_students)
