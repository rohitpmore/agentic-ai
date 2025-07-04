from pydantic import BaseModel, Field
from typing import Literal, Optional, Any
import os
from langchain_openai import ChatOpenAI
from langchain_groq import ChatGroq
from .config_loader import load_config

class ConfigLoader():
    def __init__(self):
        print("loading config")
        self.config = load_config()

    def __getitem__(self, key):
        return self.config[key]



class ModelLoader(BaseModel):
    model_provider: Literal["openai", "groq"] = "groq"
    config: Optional[ConfigLoader] = Field(default=None, exclude=True)
    def model_post_init(self, __context: Any) -> None:
        self.config = ConfigLoader()

    class Config:
        arbitrary_types_allowed = True

    def load_llm(self):
        """
        Load and return a LLM model. 
        """

        print("Loading LLM")
        print(f"Model Provider: {self.model_provider}")

        if self.model_provider == "openai":
            print("Loading OpenAI LLM")
            openai_api_key = os.getenv("OPENAI_API_KEY")
            model_name = self.config["llm"]["openai"]["model_name"]
            llm = ChatOpenAI(model=model_name, api_key=openai_api_key)
        elif self.model_provider == "groq":
            print("Loading Groq LLM")
            groq_api_key = os.getenv("GROQ_API_KEY")
            model_name = self.config["llm"]["groq"]["model_name"]
            llm = ChatGroq(model=model_name, temperature=0, api_key=groq_api_key)
        else:
            raise ValueError(f"Model provider {self.model_provider} not supported")
        
        return llm