from typing import TypedDict

class Person(TypedDict):
    name: str
    age: int
    email:str 
    bill: float
    is_online: bool

Vijay:Person = {

                 "name": "Vijay" ,
                 "age": 45 ,
                 "email": "vijay.choudhary@exam.com" ,
                 "bill": 100000 ,
                 "is_online": True 

                }

print(Vijay)














