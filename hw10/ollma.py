import ollama

model = ollama.ChatModel("llama2")  

question = "What are the benefits of using AI in healthcare?"

response = model.chat(question)

print("AI Response:", response['message'])
