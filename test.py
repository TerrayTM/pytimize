from main import LinearProgram



A = [
    [-1,2,4,2,-1],
    [0,1,0,1,1],
    [0,5,1,2,0]
]

b=[6,1,7]
c=[1,1,-1,2,3]
z= 0


p = LinearProgram(A, b, c, z)

#p.simplex_solution([1,2,3], in_place=True)

print(p.simplex_solution([1,2,3], in_place=True))
#print(p.steps_string)

# #BUG
# A = [
# [1,2,0,1],
# [1,-2,16,0],
# [8,2,3,1],
# [1,0,0,0],
# [0,0,0,1]
# ]

# b=[10,14,-2,0,0]

# c=[1,2,3,0]
# z= 0


# p = LinearProgram(A, b, c, z, inequalities=["<=", "<=", "=", ">=", ">="], free_variables=[2])

# print(p.to_sef())



# #BUG
# A = [
# [-1,-2,0,1],
# [1,2,11,0],
# [8,0,-3,1],
# [1,0,0,0],
# [0,1,0,0]
# ]

# b=[9,3,0,0,0]

# c=[1,2,3,0]
# z= 0


# p = LinearProgram(A, b, c, z, inequalities=["=", "<=", "=", ">=", ">="], free_variables=[2])
# print(p)
# print(p.to_sef())