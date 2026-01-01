from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv

load_dotenv()

my_llm = HuggingFaceEndpoint(
    model="HuggingFaceH4/zephyr-7b-beta",  # Proven hosted model
    task="text-generation"
)

model = ChatHuggingFace(llm=my_llm)

result=model.invoke("What is the capital of India")

print(result.content)



'''
# my_llm=HuggingFaceEndpoint(
#   repo_id="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
#   task="text-generation"
# )

# my_llm=HuggingFaceEndpoint(
#   repo_id="TinyLlama/TinyLlama-1.1B-step-50K-105b",
#   task="text-generation"
# )     # this isnt chat tuned so it threw an error

# my_llm=HuggingFaceEndpoint(
#   repo_id="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
#   task="text-generation"
# )

# to run locally we need pipeline
'''