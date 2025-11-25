import torch
from transformers import BartTokenizer, BartForConditionalGeneration

class BartQueryGenerator:
    def __init__(self, config):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")

        # Load model from configured path or use default pre-trained model
        model_path = config.get('paths', {}).get('model_dir', config.get('model', {}).get('bart_model', 'facebook/bart-large'))
        self.tokenizer = BartTokenizer.from_pretrained(model_path)
        self.model = BartForConditionalGeneration.from_pretrained(model_path).to(self.device)
        self.batch_size = config.get('training', {}).get('batch_size', 256)

    def format_input(self, example):
        answer = example.get('Answer', '[No Answer]')
        title = example.get('Title', '[No Title]')
        sentence = example.get('Sentence', '[No Sentence]')
        question = example.get('Question', '[No Question]')
        return f"{question} Answer: {answer} Title: {title} Sentence: {sentence}"

    def generate_expansion_batch(self, examples, max_length=10):
        inputs = [self.format_input(ex) for ex in examples]

        encoding = self.tokenizer(
            inputs,
            max_length=config["bart_max_length"],
            padding='max_length',
            truncation=True,
            return_tensors="pt"
        ).to(self.device)

        with torch.no_grad():
            outputs = self.model.generate(
                encoding.input_ids,
                max_length=max_length,
                num_beams=1,
                early_stopping=True
            )

        expansions = [self.tokenizer.decode(output, skip_special_tokens=True) for output in outputs]
        return expansions