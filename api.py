from fastapi import FastAPI, Query
app = FastAPI()


@app.get("/topics/")
async def read_item(title: str = Query("no-title",min_length=3,max_length=50)):
    return {"topic":title}

g