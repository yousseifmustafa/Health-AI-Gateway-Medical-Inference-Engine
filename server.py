from dotenv import load_dotenv
load_dotenv()

from Server.main import app

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("server:app", host="0.0.0.0", port=8000, reload=True)