
from fastapi import FastAPI

from app.routes.analyze import router as analyze_router  

app = FastAPI(title="Multimodal Trust AI")  #This line creates an instance of the FastAPI class, which is the main application object for our API. The title parameter is used to set the title of the API documentation that will be generated automatically by FastAPI. In this case, we are naming our API "Multimodal Trust AI".

app.include_router(analyze_router)    #This line includes the analyze_router into the main FastAPI application. By including the router, we are registering all the endpoints defined in the analyze_router with our main application. This allows us to organize our API routes in a modular way, keeping related routes together in separate files or modules. 
