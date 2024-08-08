


from micrograd.engine import Value

a = Value(4.0)
b = Value(3.0)
c = a * b
print(f'{c.data:.4f}')

c.backward()
print(f'{a.grad:.4f}')
print(f'{b.grad:.4f}')
print(f'{c.grad:.4f}')


a = Value(4.0)
b = Value(3.0)
d = a + b
d.backward()
print(f'{d.data:.4f}')
print(f'{d.grad:.4f}')
print(f'{a.grad:.4f}')
print(f'{b.grad:.4f}')


