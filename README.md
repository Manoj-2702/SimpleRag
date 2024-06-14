# RAG Model
This project implements a Retrieval-Augmented Generation (RAG) model using LangChain and Gradio to create a Question Answering (QA) system for PDF documents. The application leverages advanced LLMs hosted on Hugging Face to provide insightful responses based on the content of uploaded PDF documents.

## Features
- <b>Upload PDF Documents</b>: Easily upload multiple PDF files for processing.
- <b>Create Vector Database</b>: Generate a vector database from the uploaded documents for efficient information retrieval.
- <b>Select Large Language Models (LLMs)</b>: Choose from available LLMs such as Meta-Llama and Mistral for response generation.
- <b>Chat with Documents</b>: Interact with your documents through a chat interface, receiving relevant responses sourced directly from the documents.
- <b>Contextual References</b>: Obtain relevant context from the source document for each response.
- <b>Customizable Parameters</b>: Adjust LLM parameters such as temperature, max tokens, and top-k selection for fine-tuning the response generation.


## Installation
To get started, follow these steps:

1. Clone the repository:

```bash
git clone https://github.com/your-username/rag-model-athina-ai.git
cd rag-model-athina-ai
```

2. Install the required dependencies:
```bash
pip install -r requirements.txt
```

3. Set up your Hugging Face API token:

- Obtain your API token from Hugging Face.
- Set the token as an environment variable:
```bash
export HF_TOKEN='your_hugging_face_api_token'
```

4. Run the application:
```bash
python app.py
```

## Usage
### Step 1 - Upload PDF Documents and Initialize RAG Pipeline
1. Upload PDF Documents: Upload multiple PDF files for processing.
2. Create Vector Database: Click the "Create vector database" button to generate a vector database from the uploaded documents.
3. Monitor Initialization: The status of the vector database will be displayed.
### Step 2 - Initialize LLM
1. Select LLM: Choose from available LLMs using the provided radio buttons.
2. Set Parameters: Adjust LLM parameters such as temperature, max tokens, and top-k selection.
3. Initialize QA Chatbot: Click the "Initialize Question Answering Chatbot" button to set up the QA system.
### Step 3 - Chat with Your Document
1. Ask Questions: Type your questions in the input box and submit them.
2. View Responses: The chatbot will provide answers based on the content of the uploaded documents.
3. Relevant Context: View relevant context from the source document provided in the response.
### Additional Features
- Clear Chat: Use the "Clear" button to reset the chat interface.


Code Overview
Main Functions
load_doc(pdf_path): Loads and splits PDF documents into chunks.
create_db(splits): Creates a vector database using FAISS.
initialize_llmchain(llm_model, temperature, max_tokens, top_k, vector_db): Initializes the LLM chain for QA.
initialize_database(list_file_obj): Initializes the vector database.
initialize_LLM(llm_option, llm_temperature, max_tokens, top_k, vector_db): Sets up the LLM for QA.
format_chat_history(message, chat_history): Formats the chat history for display.
conversation(qa_chain, message, history): Handles the conversation logic for the chatbot.
upload_file(file_paths): Handles file upload and processing.
create_dataset(pdf_path, vector_db): Creates a dataset from PDF documents for model evaluation.
evaluate_model(dataset_path, qa_chain): Evaluates the QA model against a dataset.
Gradio Interface
The Gradio interface is set up using gr.Blocks, with components for file upload, parameter selection, and chat interaction. The interface includes callbacks for initializing the database, setting up the LLM, and handling chat conversations.

License
This project is licensed under the MIT License.

Contributing
Contributions are welcome! Please submit a pull request or open an issue to discuss your ideas.

Acknowledgments
This project leverages the following libraries and services:

LangChain
Gradio
Hugging Face
Feel free to reach out if you have any questions or need further assistance!