import sys
import os
from flask import Flask, request, jsonify
from flask_cors import CORS

# Añadir el directorio raíz al sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from model import final_result  # Ahora puedes importar final_result

def document_to_dict(document):
    return {
        "page_content": document.page_content,
        "source": document.metadata['source'],
        "page": document.metadata['page']
    }



app = Flask(__name__)
CORS(app)  # Habilitar CORS para permitir solicitudes desde el frontend

@app.route('/chat', methods=['POST'])
def chat():
    try:
        user_input = request.json['message']
        response = final_result(user_input)
        
        if 'source_documents' in response and isinstance(response['source_documents'], list):
            # Convertir cada documento a un diccionario
            response['source_documents'] = [document_to_dict(doc) for doc in response['source_documents']]
        print(response)
        return jsonify(response)
    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == '__main__':
    # Configurar el servidor para ejecutarse en el puerto 8000
    app.run(debug=True, port=5000)
