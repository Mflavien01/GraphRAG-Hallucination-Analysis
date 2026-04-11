from transformers import T5ForConditionalGeneration, T5Tokenizer

tokenizer = T5Tokenizer.from_pretrained("valhalla/t5-small-qa-qg-hl") # load the T5 tokenizer for the specified model to process input and output text
model = T5ForConditionalGeneration.from_pretrained("valhalla/t5-small-qa-qg-hl") # load the T5 model for answer extraction + question generation based on t5-small-qa-qg-hl model 

def run_model(input_text, max_length=128):
    """Run the T5 model with the given input text and return the generated output"""
    ids = tokenizer.encode(input_text, return_tensors="pt", max_length=512, truncation=True) # tokenize the input text to be processed by the model
    output = model.generate(ids, max_length, num_beams=4, early_stopping=True) #apply the model 
    return tokenizer.decode(output[0], skip_special_tokens=True) #detokenize the output to get the generated text

def extract_answers(text):
    answers = []
    for sent in text.split("."): # split sentence in every dot to obtain individual sentences
        sent = sent.strip()
        # model splits string with <sep> to obtain answers individually
        raw = run_model(f"extract_answers: <hl> {sent} <hl>") # model return answers spans splitted by <sep>
        for a in raw.split("<sep>"):
            a = a.strip() #remove whitespace of answers
            answers.append(a)
    return list(set(answers))

def generate_question(text, answer):
    highlighted = text.replace(answer, f"<hl> {answer} <hl>", 1) # highlight the answer in the sentence to be processed by the model for question generation
    return run_model(f"generate question: {highlighted}", max_length=64) # model return the generated question based on the highlighted answer in the sentence


with open("input_text.txt", "r", encoding="utf-8") as f:
    sentences = [line.strip() for line in f if line.strip()]

for i, sentence in enumerate(sentences):
    print(f"\n[{i+1}] {sentence}")
    for ans in extract_answers(sentence):
        q = generate_question(sentence, ans)
        print(f"  Q: {q}")
        print(f"  A: {ans}")