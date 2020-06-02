"""Tutorial for the use of inheritance and super() function

* Initializing all attributes defined by the parent classes is NOT imperative
* Attributes can only be overwritten by the child class
* Methods can both be overwritten, but also overloaded through the use of super method

* super(class_name, object) allows the call of methods of the parent of class_name
This is the only way, to call a method of the parent class that has been overloaded by the child class.

"""


class Rectangle:
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def area(self):
        return self.x * self.y

    def perimeter(self):
        return self.x + self.y


class Square(Rectangle):
    def __init__(self, x):
        super().__init__(x, x)
        self.x = 5

    def area(self):
        return self.x**2

    def perimeter(self):
        return self.x*2

    def area2(self):
        return super().area()


# class Doubler(Rectangle):
#     def double_area(self):
#         return 2*(self.x + self.y)
#
#
# class Triangle:
#     def __init__(self, base, height):
#         self.base = base
#         self.height = height
#
#     def area(self):
#         return (self.base*self.height)/2
#
#
#
# class HouseFacade(Triangle, Square):
#     def __init__(self, x, height):
#         self.aek1 = x
#         self.aek2 = height
#         super().__init__(x, height)
#         super(HouseFacade, s)

rect = Rectangle(3, 2)
square = Square(3)
# triangle = Triangle(2, 3)
# house_facade = HouseFacade(3, 3)
print(rect.area())
print(square.area())
# print(triangle.area())