from pytimize.formulations.linear import maximize, x

p = maximize(x[1] + x[2]).subject_to(
    4 * x[1] + x[2] <= 4,
    x[1] + x[2] - 1 == 0,
    10 * x[2] <= 15 + 3 * x[1]
).where(
    x >= 0
)

print(p)
print(f"Solution: {p.solve()}")
