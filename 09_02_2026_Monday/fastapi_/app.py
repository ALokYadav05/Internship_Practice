from fastapi import FastAPI
from pydantic import BaseModel, Field
from typing import Optional

app = FastAPI()

class Task(BaseModel):
    id : int = Field(...,ge=0,description="Task id")
    description : str = Field(...,description="Task description")
    due_date : Optional[str] = Field(default=None, description="Task-due_date")
    status : str = Field(default='pending', description="Task status", examples=["pending", "completed"])

task_db = [{
    'id': 1,
    'description' : 'Task1',
    'due_date' : '09-02-2026',
    'status' : 'pending'
}]


@app.get('/task', tags=['GET'])
def get_all_task():
    return task_db
    
@app.post('/task', tags=['POST'])
def add_task(task: Task):
    task_dict = task.model_dump(exclude_unset=True)  # we convert pydantic object task to simple dictionary,
                                                 # bcz when we write task:Task , the data comes object.
                                                 # (exclude_unset=True) -> ensure that only take values that user define
    task_db.append(task_dict)
    return task_db


