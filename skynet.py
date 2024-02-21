from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

from record import voice2text

from text2speech import text2speech

import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"


tokenizer = AutoTokenizer.from_pretrained("microsoft/DialoGPT-medium", padding_side="left")
model = AutoModelForCausalLM.from_pretrained("microsoft/DialoGPT-medium")

# Let's chat for 5 lines

chat_history_ids = None  # Inicializa el historial de chat vacío

#for step in range(5):
while True:

    prompt = voice2text()

    # encode the new user input, add the eos_token and return a tensor in Pytorch
    #new_user_input_ids = tokenizer.encode(input(">> User:") + tokenizer.eos_token, return_tensors='pt')

    print(">> User: "+ prompt)

    new_user_input_ids = tokenizer.encode(prompt + tokenizer.eos_token, return_tensors='pt')

    # append the new user input tokens to the chat history
    #bot_input_ids = torch.cat([chat_history_ids, new_user_input_ids], dim=-1) if step > 0 else new_user_input_ids
    bot_input_ids = torch.cat([chat_history_ids, new_user_input_ids], dim=-1) if chat_history_ids is not None else new_user_input_ids
    # generated a response while limiting the total chat history to 1000 tokens, 
    chat_history_ids = model.generate(bot_input_ids, max_length=1000, pad_token_id=tokenizer.eos_token_id)

    # pretty print last ouput tokens from bot
    answer = tokenizer.decode(chat_history_ids[:, bot_input_ids.shape[-1]:][0], skip_special_tokens=True)
    print("Skynet: {}".format(answer))
    text2speech(answer,"en")
    input()
