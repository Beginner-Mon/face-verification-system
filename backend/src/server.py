from fastapi import FastAPI
from router.router import router 
from fastapi.middleware.cors import CORSMiddleware
# Create the main FastAPI app
app = FastAPI(title  = "Model Prediction API",)
# CORS configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://localhost:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Include the router with a prefix
app.include_router(
    router,
    prefix="/api",
    tags=["predictions"]
)

# Root endpoint
@app.get("/")
async def root():
    return {"message": "Welcome to the Machine Learning API"}