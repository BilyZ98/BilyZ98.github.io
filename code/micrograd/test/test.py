
from micrograd2.engine  import Value



# a = Value(4.0)
# b = Value(3.0)
# c = a * b


# print(f'{c.data:.4f}')

# c.backward()
# print(f'{a.grad:.4f}')
# print(f'{b.grad:.4f}')
# print(f'{c.grad:.4f}')


# d = Value(5.0)
# e = d ** 2
# print(f'{e.data:.4f}')
# e.backward()
# print(f'{d.grad:.4f}')



f = Value(4.0)
g = Value(2.0)
h = f/g

h.backward()
print(f'{f.grad:.4f}')
print(f'{g.grad:.4f}')


