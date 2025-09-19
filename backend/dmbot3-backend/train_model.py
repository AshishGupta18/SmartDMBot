import json
import joblib
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
import os

def flatten_rules(rules):
    data = []
    objects = rules.get('objects', {})
    for obj_type, obj_data in objects.items():
        for status, answer in obj_data.items():
            if status == 'follow_up_question':
                continue  # skip follow-up question
            
            # Handle TABL.NEW case - it's a list, not a dict
            if obj_type == 'TABL' and status == 'NEW' and isinstance(answer, list):
                # For TABL.NEW, we need to handle the special case where we have segments
                # Since the current structure doesn't have segments, we'll treat each item as a separate training example
                for i, item in enumerate(answer):
                    inp = f"{obj_type} | {status} | segment_{i+1}"
                    data.append((inp, item))
            elif isinstance(answer, list):
                # For other cases where answer is a list, join them into a single string
                inp = f"{obj_type} | {status}"
                data.append((inp, " ".join(answer)))
            else:
                # For string answers
                inp = f"{obj_type} | {status}"
                data.append((inp, answer))
    return data

def main():
    with open('rules.json', 'r', encoding='utf-8') as f:
        rules = json.load(f)
    data = flatten_rules(rules)
    X, y = zip(*data)
    
    # Convert y to a list to ensure it's 1D
    y = list(y)
    
    # Build pipeline
    clf = Pipeline([
        ('tfidf', TfidfVectorizer()),
        ('lr', LogisticRegression(max_iter=1000))
    ])
    clf.fit(X, y)
    joblib.dump(clf, 'chatbot_model.pkl')
    print('Model trained and saved as chatbot_model.pkl')
    print(f'Training data shape: X={len(X)}, y={len(y)}')

if __name__ == '__main__':
    main()
