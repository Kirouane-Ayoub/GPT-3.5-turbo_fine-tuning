import openai
import os
os.environ["OPENAI_API_KEY"] = ""
file_name = ""
n_epochs = 100
model="gpt-3.5-turbo"
def train(file_name , n_epochs , model ) :
  upload_response = openai.File.create(
    file=open(file_name, "rb"),
    purpose='fine-tune'
  )
  training_file_id = upload_response.id
  create_args = {
	"training_file": training_file_id,
	"model": model,
	"n_epochs": n_epochs,
	"batch_size": 2,
	"learning_rate_multiplier": 0.3
  }
  fine_tune_response = openai.FineTune.create(**create_args)
  job_id = fine_tune_response["id"]
  status = fine_tune_response["status"]
  return fine_tune_response , job_id , status , training_file_id
