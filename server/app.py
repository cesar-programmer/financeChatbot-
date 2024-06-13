import sys
import os
from flask import Flask, request, jsonify
from flask_cors import CORS

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from model import final_result

def document_to_dict(document):
    return {
        "page_content": document.page_content,
        "source": document.metadata['source'],
        "page": document.metadata['page']
    }



app = Flask(__name__)
CORS(app)

@app.route('/chat', methods=['POST'])
def chat():
    try:
        user_input = request.json['message']
        response = final_result(user_input)
        
        if 'source_documents' in response and isinstance(response['source_documents'], list):
            response['source_documents'] = [document_to_dict(doc) for doc in response['source_documents']]
        print(response)
        return jsonify(response)
    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == '__main__':
    app.run(debug=True, port=5000)
