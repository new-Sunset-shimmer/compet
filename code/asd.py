from transformers import AutoTokenizer, AutoModel,AdamW,get_linear_schedule_with_warmup,AutoModelForSequenceClassification,AutoImageProcessor, ResNetForImageClassification
tokenizer = AutoTokenizer.from_pretrained('microsoft/deberta-v3-large',do_lower_case=True)

text1 = "Tried to allocate"
text2 = "reserved in total"

output = tokenizer.encode_plus(text1, text2)

print(output)