import json
import os
import asyncio
from dotenv import load_dotenv
from crewai_tools import CodeDocsSearchTool
from crewai import Agent, Task, Crew, Process
from langchain_google_genai import ChatGoogleGenerativeAI
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import uvicorn
from fastapi.middleware.cors import CORSMiddleware

# Load environment variables
load_dotenv()

# Configure the FastAPI app
app = FastAPI()

# Allow all origins (CORS anywhere)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows any domain
    allow_credentials=True,
    allow_methods=["*"],  # Allows all HTTP methods
    allow_headers=["*"],  # Allows all headers
)

# Configure the CodeDocsSearchTool for your documentation site with Google embeddings
docs_tool = CodeDocsSearchTool(
    docs_url='https://react-nex-docs.vercel.app/',
    config=dict(
        embedder=dict(
            provider="google",
            config=dict(model="models/embedding-001", task_type="retrieval_document"),
        )
    )
)

# Function to initialize the LLM asynchronously
async def initialize_llm():
    llm = ChatGoogleGenerativeAI(
        model="gemini-1.5-flash",
        verbose=True,
        temperature=0.5,
        google_api_key=os.getenv("GOOGLE_API_KEY")
    )
    return llm

# Create the MERN Developer agent with CodeDocsSearchTool
async def create_agent(llm):
    mern_dev_agent = Agent(
        role="Professional MERN Developer",
        goal="Generate optimized and accurate code snippets in response to {prompt}",
        memory=True,
        verbose=True,
        backstory=(
            "An experienced MERN developer proficient in React.js and Tailwind CSS,"
            "creating tailored solutions by leveraging the provided documentation."
        ),
        tools=[docs_tool],
        llm=llm
    )
    return mern_dev_agent

# Define a code generation task with an emphasis on using custom components
async def create_task(mern_dev_agent):
    code_generation_task = Task(
        description=(
            "Retrieve relevant data from the documentation site and generate code for {prompt}."
            "Ensure that the code follows best practices in React.js and Tailwind CSS, strictly using"
            "the components from the React-Nex library (e.g., <Button> instead of native HTML elements)."
            "1. Import components from the React-Nex library using the pattern:\n"
            "   import ComponentName from './component-react-nex/ComponentName/ComponentName'\n"
            "2. Each component is located in its own folder under component-react-nex:\n"
            "   - Single component example: './component-react-nex/Button/Button'\n"
            "   - Related components example: './component-react-nex/Modal/Modal'\n\n"
        ),
        expected_output="Code snippet using React-Nex components for {prompt}.",
        tools=[docs_tool],
        agent=mern_dev_agent,
    )
    return code_generation_task

# Configure the Crew to execute the process sequentially
async def create_crew(mern_dev_agent, code_generation_task):
    crew = Crew(
        agents=[mern_dev_agent],
        tasks=[code_generation_task],
        process=Process.sequential,
    )
    return crew

# Define request model for API
class PromptRequest(BaseModel):
    prompt: str

# Endpoint to handle code generation based on prompt
@app.post("/generate-code")
async def generate_code(request: PromptRequest):
    try:
        # Initialize LLM and agent
        llm = await initialize_llm()
        mern_dev_agent = await create_agent(llm)
        code_generation_task = await create_task(mern_dev_agent)
        crew = await create_crew(mern_dev_agent, code_generation_task)

        # Execute the task with the user prompt
        result = crew.kickoff(inputs={'prompt': request.prompt})

        if result:
            # Strip formatting characters from the result and return as JSON
            cleaned_code = result.strip("```javascript").strip("```")
            return {"prompt": request.prompt, "generated_code": result}
        else:
            raise HTTPException(status_code=500, detail="Unexpected result format")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Root endpoint to verify server status
@app.get("/")
async def root():
    return {"message": "Welcome to the Code Generation API! Use /generate-code to generate code based on a prompt."}


# Add this in your script to run the app on the correct host and port
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=int(os.getenv("PORT", 8000)))