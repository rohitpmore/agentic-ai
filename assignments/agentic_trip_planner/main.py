from fastapi import FastAPI
from pydantic import BaseModel
from agent.agentic_workflow import GraphBuilder
from fastapi.responses import JSONResponse
import os

app = FastAPI()

class QueryRequest(BaseModel):
    query: str

@app.post("/query")
async def query_travel_agent(query: QueryRequest):
    try:
        print(f"Received query: {query}")
        graph = GraphBuilder(model_provider="groq")
        react_app = graph()

        png_graph = react_app.get_graph().draw_mermaid_png()
        with open("graph.png", "wb") as f:
            f.write(png_graph)

        print(f"Graph saved as 'my_graph.png' in {os.getcwd()}")
        
        messages = {"messages": [query.question]}
        output = react_app.invoke(messages)

        if isinstance(output, dict) and "messages" in output:
            final_output = output["messages"][-1].content
        else:
            final_output = str(output)
        
        return {"answer": final_output}

    except Exception as e:
        return JSONResponse(status_code=500,content={"error": str(e)})




def main(): 
    print("Hello from agentic-trip-planner!")


if __name__ == "__main__":
    main()
