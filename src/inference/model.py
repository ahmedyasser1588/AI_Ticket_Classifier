from transformers import AutoTokenizer, AutoModelForCausalLM
from .config import Model_Name




def load_model() : 
    tokenizer = AutoTokenizer.from_pretrained(Model_Name)
    model     = AutoModelForCausalLM.from_pretrained(Model_Name,device_map='auto',torch_dtype='auto')
    return tokenizer, model


def get_model_device(model) :
    return model.get_input_embeddings().weight.device
