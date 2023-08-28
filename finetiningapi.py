from fastapi import FastAPI, File, UploadFile, Form
import openai
import os

app = FastAPI()

@app.post("/train")
async def train(
    file: UploadFile = File(...),
    n_epochs: int = Form(...),
):
    try:
        os.environ["OPENAI_API_KEY"] = ""
        openai.api_key = os.getenv("OPENAI_API_KEY")

        upload_response = openai.File.create(
            file=file.file,
            purpose='fine-tune'
        )
        training_file_id = upload_response.id
        
        create_args = {
            "training_file": training_file_id,
            "model": "gpt-3.5-turbo",
            "n_epochs": n_epochs,
            "batch_size": 2,
            "learning_rate_multiplier": 0.3
        }
        
        fine_tune_response = openai.FineTune.create(**create_args)
        job_id = fine_tune_response["id"]
        status = fine_tune_response["status"]
        
        return {
            "fine_tune_response": fine_tune_response,
            "job_id": job_id,
            "status": status,
            "training_file_id": training_file_id
        }

    except Exception as e:
        return {"error": str(e)}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
