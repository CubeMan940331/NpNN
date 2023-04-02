import network, hashlib
import matplotlib.pyplot as plt
import load_mnist as lm

dataset=lm.load_mnist()
railgun=network.net('test')
print(railgun.shape_list)
print('interactive mode')
print(railgun.accuracy(dataset['x_test'],dataset['y_test']))
while True:
    tmp=input('enter a number : ')
    try:tmp=int(tmp)
    except ValueError:
        sha3_256=hashlib.sha3_256()
        sha3_256.update(tmp.encode())
        tmp=int.from_bytes(sha3_256.digest(),byteorder='big')
    n=tmp%10000
    print("idx =",n)
    railgun.calculate(dataset['x_test'][n])
    print('network : ',network.echo_ans(railgun.a_list[-1]))
    print('ans : ',network.echo_ans(dataset['y_test'][n]))
    plt.imshow(dataset['x_test'][n].reshape(28,28),cmap='gray')
    plt.show()

