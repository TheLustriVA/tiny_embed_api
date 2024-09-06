from transformers import BertModel, BertTokenizer
import torch

# Load the pre-trained model and tokenizer
model_name = "bert-base-uncased"
tokenizer = BertTokenizer.from_pretrained(model_name)
model = BertModel.from_pretrained(model_name)

def generate_embedding(input_text):
    # Encode the input text
    encoded_input = tokenizer(input_text, return_tensors="pt")

    # Generate the embedding
    with torch.no_grad():
        output = model(**encoded_input)
        embedding = output[0].mean(dim=0).squeeze()

    return embedding.tolist()
