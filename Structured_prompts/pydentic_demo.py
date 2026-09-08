from pydantic import BaseModel, EmailStr 
from typing import List, Optional 

class Student(BaseModel):
    name: str 
    email: EmailStr

new_student =  {"name" : "Abhijeet", "email": "withlove@google.com"}

student  =  Student(**new_student)

print(student)