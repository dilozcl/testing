# -*- coding: utf-8 -*-
"""
Created on Sat Feb 18 00:04:15 2023

@author: Diloz
"""

import numpy as np

"""
The classes are used to create objects
Objects have attributes and methods
Attributes represent data about the object
Methods represent functionality that the object can do
"""

class fruit:
    def __init__(self, name, color):
        
        self.name = name
        self.color = color
        
my_fruit = fruit("apple", "red")
your_fruit =  fruit("mango", "yellow")
print("my_fruit is", my_fruit.name, my_fruit.color)

print("your_fruit is", your_fruit.name, your_fruit.color)



# Method
class data:
    def __init__(self, alphanum):
        self.alphanum = alphanum

    def tobinary(self):
        binary = int("".join([bin((ord(c)))[2:] for c in self.alphanum]))
        return binary

d1 = data("helloworld")

binar= d1.tobinary()


