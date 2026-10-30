from openai import OpenAI


class LLM:
    def __init__(self,base_url,api_key):
        self.client=OpenAI(base_url=base_url,api_key=api_key)
    def generate(self,prompt,model):
        res=self.client.responses.create(model=model,input=prompt)
        return res.output_text