from transformers import AutoTokenizer, AutoModel

#Using FinBERT for Finnish text
model_name = "TurkuNLP/bert-base-finnish-cased-v1"

#Load tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_name)

#Legislative text here
text = "Suomen alue on jakamaton. Valtakunnan rajoja ei voida muuttaa ilman eduskunnan suostumusta."

#Tokenizing
tokens = tokenizer.tokenize(text)
print("Tokens: ", tokens)

#Get the IDs
token_ids = tokenizer.encode(text)
print("Token IDs:", token_ids)

# Decode back to check
decoded = tokenizer.decode(token_ids)
print("Decoded:", decoded)