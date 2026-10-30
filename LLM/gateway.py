from LLM import config
from .base import LLM
from .Model import GenerationRequest 
from .exceptions import (LLMError,ConnectionError,ProvideError)
import time                    
import random                                   
import logging

logger=logging.getLogger(__name__)

class LLMGateWay(LLM):
    def __init__(self,llm:LLM,max_attempts:int=3,base_delay:float=1,max_delay:float=30.0):

        self.llm=llm
        self.max_attempts=max_attempts
        self.base_delay=base_delay
        self.max_delay=max_delay

        if max_attempts<1:
            raise ValueError("max_attempts must be greater than 0")
        if base_delay<0:
            raise ValueError("base_delay must be greater than 0")
        if max_delay<0:
            raise ValueError("max_delay must be greater than 0")
        if max_delay<base_delay:
            raise ValueError("max_delay must be greater than or equal to base_delay")

    logger.info("Gateway initialized...........")

    def generate(self,request:GenerationRequest):
        for attemp in range(1,self.max_attempts+1):
            try: 
                return self.llm.generate(request)
                logger.info("Request Finished Sucesfully")
            except (ConnectionError,ProvideError):
               delay=min(self.max_delay,random.uniform(self.base_delay * 2**(attemp-1),0.0))
               logger.error("Trying again after delay")    
               logger.error(f"Attempt: {attemp}")
               time.sleep(delay)
               

            except LLMError as e:
                logger.critical("Exiting because of :",e)
                raise
                
        logger.critical(f"Failed to get response after {self.max_attempts} attempts")
        raise LLMError("Too many attempts") 
            
            
    
