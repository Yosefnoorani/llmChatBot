from flask import Flask, render_template, request, jsonify, redirect, url_for
import generate_embeddings  # Assuming this module exists
import answer_query_with_streaming
import os
from werkzeug.utils import secure_filename
from answer_query_with_streaming import query_embedding_with_streaming
from flask import Response, stream_with_context


app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/embed_files', methods=['POST'])
def embed_files():
    try:
        # Ensure temp_uploads directory exists
        os.makedirs('temp_uploads', exist_ok=True)

        # Get uploaded files
        uploaded_files = request.files.getlist('files')

        # Save files temporarily
        file_paths = []
        for file in uploaded_files:
            # Sanitize filename to prevent path traversal
            safe_filename = secure_filename(file.filename)
            temp_path = os.path.join('temp_uploads', safe_filename)
            file.save(temp_path)
            file_paths.append(temp_path)

        # Get other form data
        model_name = request.form.get('model_name')
        chunk_size = int(request.form.get('chunk_size'))
        overlap = int(request.form.get('overlap'))

        file_paths = './temp_uploads'

        # Call embedding function
        result_file = generate_embeddings.embedding_files_multiple_dirs(
            file_paths, model_name, chunk_size, overlap
        )

        return jsonify({
            'status': 'success',
            'result_file': result_file
        })
    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500

@app.route('/chat', methods=['GET'])
def chat():
    # Get the JSON file path from the query parameters
    json_file = request.args.get('json_file', '')
    return render_template('chat.html', json_file=json_file)


@app.route('/query', methods=['POST'])
def handle_query():
    data = request.json
    query = data.get('query')
    language_model = data.get('language_model')
    embedding_model = data.get('embedding_model')
    json_file = data.get('json_file')

    def generate():
        try:
            answer_query_with_streaming.initialize_settings(language_model, embedding_model)
            nodes = answer_query_with_streaming.load_embeddings_from_json(json_file, embedding_model)

            if nodes is None:
                yield "No embeddings found for the selected model."
                return

            stream_generator = answer_query_with_streaming.build_and_query_index(nodes, query, top_k=5)

            for chunk in stream_generator:
                yield chunk
        except Exception as e:
            yield f"Error processing query: {str(e)}"

    return Response(stream_with_context(generate()), content_type='text/plain')


if __name__ == '__main__':
    app.run(debug=True)
