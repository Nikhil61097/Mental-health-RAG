import json
from bs4 import BeautifulSoup  # pip install beautifulsoup4

def parse_nq_line(json_line):
    data = json.loads(json_line)
    
    question = data.get('question_text', '')
    # Prefer plain text if available, else clean HTML
    if 'document_text' in data:
        context = data['document_text']
    else:
        html = data.get('document_html', '')
        context = BeautifulSoup(html, 'html.parser').get_text(separator=' ', strip=True)
    
    annotations = data.get('annotations', [])
    answer_texts = []
    if annotations:
        short_answers = annotations[0].get('short_answers', [])
        tokens = context.split()
        for ans in short_answers:
            start = ans['start_token']
            end = ans['end_token']
            if 0 <= start < end <= len(tokens):
                answer_texts.append(' '.join(tokens[start:end]))
    answer = '; '.join(answer_texts) if answer_texts else ''
    
    return {
        'question': question,
        'answer': answer,
        'context': context
    }

def extract_nq_subset(input_file, output_file, num_samples=100):
    with open(input_file, 'r', encoding='utf-8', errors='replace') as f_in, \
         open(output_file, 'w', encoding='utf-8') as f_out:
        count = 0
        for line in f_in:
            if count >= num_samples:
                break
            parsed = parse_nq_line(line)
            json.dump(parsed, f_out)
            f_out.write('\n')
            count += 1
    print(f"Extracted {count} samples to {output_file}")

if __name__ == "__main__":
    input_path = r"C:\Users\nchha\Desktop\OpenAI-training\Nq.jsonl"  # Update this path
    output_path = r"C:\Users\nchha\Desktop\OpenAI-training\nq_subset.jsonl"  # Path for smaller subset
    
    extract_nq_subset(input_path, output_path, num_samples=100)
# sample usage