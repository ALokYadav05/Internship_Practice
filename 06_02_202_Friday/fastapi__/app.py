from fastapi import FastAPI
from pydantic import BaseModel
from typing import Optional


app = FastAPI()

student = {
    1: {
        "name": "Alok",
        "age": 21,
        "height": 187.5
    }
}

class Students(BaseModel):
    name: Optional[str] = None
    age: Optional[int] = None
    height: Optional[float] = None

class Students2(BaseModel):
    name: str
    age: int
    height: float


@app.get("/get/{id_}", tags=["Students"])
async def get_item(id_: int):
    if id_ not in student:
        return {"message": "Student not found"}
    return student[id_]


@app.post("/items/post")
async def add_student(data:Students):
    _id = len(student) + 1
    student[_id] = data.to_dict()
    return {"message": "Student added successfully", "student": student[_id]}


@app.put("/items/{id_}")
async def update_item(id_: int, data:Students2):
    if id_ in student:
        student[id_] = data.to_dict()
        return {"message": "Student updated successfully", "student": student[id_]}
    return None


@app.patch("/items/{id_}")
async def update_item1(id_: int, data:Students):
    if id_ in student:
        if data.name:
            student[id_]["name"] = data.name
        if data.age:
            student[id_]["age"] = int(data.age)
        if data.height:
            student[id_]["height"] = float(data.height)
        return {"message": "Student updated successfully", "student": student[id_]}
    else:
        return {"message": "Student not found"}


@app.delete("/items/{_id}")
def delete_item(_id:int):
    if _id in student:
        var1 = student.pop(_id)
        return {"message": "Student deleted successfully", "student": var1}
    else:
        return {'message' : 'student not found!'}