import os 
from dotenv import load_dotenv
load_dotenv()

Model_Name = os.getenv("Model") 
DataSet = os.getenv("DataSet") 

def Print_ModelName():
    print(Model_Name)
Print_ModelName()

