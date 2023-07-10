from flask import Flask, request, jsonify
from transformers import BartTokenizer, BartForConditionalGeneration
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

class Summarizer:
  def __init__(self, pretrained_model):
    self.model = BartForConditionalGeneration.from_pretrained(pretrained_model)
    self.tokenizer = BartTokenizer.from_pretrained(pretrained_model)

  def count_tokens(self, content):
    content = " ".join(content.split())
    tokens = content.split()
    return len(tokens)

  def summarize_content(self, content):
    n_tokens = self.count_tokens(content)
    min_length = int(n_tokens * 0.4)
    max_length = int(n_tokens * 0.5)

    inputs = self.tokenizer([content], max_length=2048, truncation=True, return_tensors="pt")
    summary_ids = self.model.generate(inputs["input_ids"], min_length=min_length, max_length=max_length)
    summarized_content = self.tokenizer.batch_decode(summary_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
    return summarized_content

pretrained_model = "facebook/bart-large-cnn"
summarizer = Summarizer(pretrained_model)

# Định nghĩa route và xử lý yêu cầu POST
@app.route('/summarize', methods=['POST'])
def handle_post_request():
    json_data = request.get_json()  # Lấy dữ liệu JSON từ yêu cầu POST

    if json_data and 'data' in json_data:
        data = json_data['data']
        summarized_content = summarizer.summarize_content(data)

        # Trả về phản hồi dưới dạng JSON
        return jsonify({'result': summarized_content})
    else:
        return jsonify({'error': 'Invalid JSON data or missing field'})

# Chạy ứng dụng trên localhost với cổng 8000
if __name__ == '__main__':
    app.run('localhost', 8000)
