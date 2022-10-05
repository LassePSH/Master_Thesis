# generate random number

import random



def get_x(x):
    return x+1+random.randint(0, 100)


def main(x):
    y=get_x(x)
    print(y)