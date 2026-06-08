# RAG Chatbot 🤖

An intelligent Retrieval Augmented Generation (RAG) chatbot that answers questions based on a custom knowledge base using advanced NLP techniques. The chatbot uses semantic search with FAISS and generates answers using a fine-tuned language model.

---

## 🌟 Features

- **Retrieval Augmented Generation (RAG)**: Answers questions only from internal documents for accurate, grounded responses
- **FAISS Vector Database**: Efficient semantic similarity search for fast document retrieval
- **Sentence Embeddings**: Uses state-of-the-art sentence transformers for semantic understanding
- **Lightweight LLM**: Google FLAN-T5 model for fast inference and accurate text generation
- **Streamlit UI**: Clean, interactive web interface for easy interaction
- **Latency Tracking**: Real-time response time monitoring
- **Context Transparency**: Shows retrieved context documents used to generate answers
- **Jupyter Notebook**: Complete pipeline walkthrough for experimentation and learning
- **CPU-Friendly**: Works efficiently on CPU without GPU requirements

---

## 🛠️ Tech Stack

### Core Libraries
- **Streamlit** - Interactive web framework for the UI
- **Sentence Transformers** - State-of-the-art sentence embedding model (`all-MiniLM-L6-v2`)
- **FAISS** - Facebook AI Similarity Search for efficient vector similarity
- **Transformers** - Hugging Face transformers (Google FLAN-T5 small model)
- **NumPy** - Numerical computing and array operations

### Additional Tools
- **Python 3.8+** - Programming language
- **Jupyter Notebook** - For experimentation and pipeline walkthrough

---

## 📋 Prerequisites

Before you begin, ensure you have the following installed:
- **Python 3.8 or higher**
- **pip** (Python package manager)
- Basic understanding of RAG concepts (optional)

---

## 🚀 Installation & Setup

### 1. Clone the Repository
```bash
git clone https://github.com/guruneela385/rag_chatbot.git
cd rag_chatbot
```

### 2. Create a Virtual Environment (Recommended)
```bash
# Using venv
python -m venv venv

# Activate virtual environment
# On Windows:
venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

**Or install manually:**
```bash
pip install streamlit
pip install sentence-transformers
pip install transformers
pip install faiss-cpu
pip install torch
pip install numpy
```

### 4. Prepare Document Data

Create a `documents` folder in the root directory and add your `.txt` files:

```
rag_chatbot/
├── documents/
│   ├── document1.txt
│   ├── document2.txt
│   ├── document3.txt
│   └── ...
├── app.py
├── rag_pipeline.ipynb
└── README.md
```

Each `.txt` file should contain relevant knowledge for your chatbot. Lines are automatically split and processed.

### 5. Run the Streamlit App
```bash
streamlit run app.py
```

The app will open in your default browser at `http://localhost:8501`

---

## 📁 Project Structure

```
rag_chatbot/
├── app.py                      # Main Streamlit application
├── rag_pipeline.ipynb          # Jupyter notebook with complete pipeline
├── documents/                  # Knowledge base directory
│   ├── document1.txt
│   ├── document2.txt
│   └── ...
├── requirements.txt            # Python dependencies
└── README.md                   # This file
```

### File Descriptions

- **app.py**: Main Streamlit application with the following components:
  - Model loading and caching
  - Document retrieval system
  - Answer generation pipeline
  - Interactive UI with chat interface

- **rag_pipeline.ipynb**: Jupyter notebook containing:
  - Complete RAG pipeline walkthrough
  - Document loading process
  - Embedding generation
  - FAISS index creation
  - Query and answer generation
  - Example queries and results

- **documents/**: Directory containing knowledge base text files

---

## 🔄 How It Works

### 1. **Document Loading**
- Reads all `.txt` files from the `documents/` folder
- Splits documents into lines for granular retrieval

### 2. **Embedding Generation**
- Uses `sentence-transformers` (all-MiniLM-L6-v2 model)
- Converts each document line into a 384-dimensional vector
- Captures semantic meaning of text

### 3. **FAISS Indexing**
- Creates efficient vector index for similarity search
- Stores embeddings in L2 (Euclidean distance) index
- Enables fast retrieval of relevant documents

### 4. **Query Processing**
- Converts user query to embedding using same model
- Retrieves top-k most similar documents using FAISS
- Ranks results by semantic similarity

### 5. **Answer Generation**
- Concatenates retrieved documents as context
- Creates prompt with RAG instruction
- Uses Google FLAN-T5 small model to generate answer
- Returns answer based only on provided context

### 6. **Response Display**
- Shows generated answer
- Displays retrieved context for transparency
- Reports latency for performance monitoring

---

## 💻 Usage Examples

### Running the Streamlit App

```bash
streamlit run app.py
```

Then ask questions like:
- "What is RAG?"
- "How does Retrieval Augmented Generation work?"
- "What is FAISS?"
- "What are sentence embeddings?"

### Running the Jupyter Notebook

```bash
jupyter notebook rag_pipeline.ipynb
```

Execute cells sequentially to:
1. Load and preprocess documents
2. Generate embeddings
3. Build FAISS index
4. Test queries
5. Analyze results and latency

---

## 🔧 Configuration

### Model Configuration

You can customize the models used in `app.py`:

```python
# Change embedding model
embed_model = SentenceTransformer("all-mpnet-base-v2")  # Larger, more accurate

# Change LLM
llm = pipeline(
    "text2text-generation",
    model="google/flan-t5-base",  # Larger T5 model
    max_new_tokens=200,
    device=-1  # -1 for CPU, 0 for GPU
)
```

### Retrieval Parameters

Adjust retrieval behavior in the `retrieve()` function:

```python
def retrieve(query, top_k=5):  # Increase top_k for more context
    query_embedding = embed_model.encode([query])
    _, indices = index.search(query_embedding, top_k)
    return [docs[i] for i in indices[0]]
```

---

## 📊 Performance Metrics

### Typical Performance (CPU)
- **Embedding Generation**: ~0.05s per query
- **Document Retrieval**: ~0.01s for FAISS search
- **Answer Generation**: ~2-5s depending on query complexity
- **Total Latency**: ~2-6 seconds

### Memory Usage
- **Embedding Model**: ~60MB
- **LLM Model**: ~200MB
- **FAISS Index**: ~10MB per 1000 documents

---

## 🎯 Example Output

```
Question: "What is FAISS?"

Answer:
"a vector database library for efficient similarity search"

Retrieved Context:
"FAISS is a library for efficient similarity search and clustering of dense vectors."
"It is developed by Facebook AI Research."
"FAISS provides implementations for similarity search indexes."

Response generated in 3.45 seconds
```

---

## 🔐 Important Notes

- **Knowledge Base Limitation**: The chatbot only answers questions based on documents in the `documents/` folder
- **Context Dependency**: Response quality depends on document quality and relevance
- **No Internet Access**: All answers come from local documents, no web search
- **Privacy**: All processing happens locally, no data sent to external servers

---

## 🚀 Future Enhancements

- [ ] Support for PDF, DOCX, and other document formats
- [ ] Real-time document addition without restart
- [ ] Multi-language support
- [ ] Caching mechanisms for faster responses
- [ ] Advanced query expansion techniques
- [ ] Document chunking strategies (sliding window, semantic chunks)
- [ ] Conversation memory and context tracking
- [ ] Custom fine-tuned models
- [ ] REST API for integration
- [ ] Docker containerization

---

## 🐛 Troubleshooting

### Issue: Models downloading very slowly
**Solution**: Pre-download models:
```python
from sentence_transformers import SentenceTransformer
SentenceTransformer("all-MiniLM-L6-v2")  # Downloads and caches
```

### Issue: "No module named 'faiss'" or "'sentence_transformers'"
**Solution**: Install missing dependencies:
```bash
pip install faiss-cpu sentence-transformers transformers
```

### Issue: Out of memory errors
**Solution**: Use smaller models:
```python
# Use smaller sentence transformer
embed_model = SentenceTransformer("all-MiniLM-L6-v2")

# Use smaller LLM
model="google/flan-t5-small"  # Instead of base or large
```

### Issue: Streamlit not opening
**Solution**: Try:
```bash
streamlit run app.py --logger.level=debug
```

---

## 📚 Learning Resources

- [RAG Explained](https://python.langchain.com/docs/use_cases/question_answering/)
- [FAISS Documentation](https://github.com/facebookresearch/faiss/wiki)
- [Sentence Transformers](https://www.sbert.net/)
- [Google FLAN-T5](https://huggingface.co/docs/transformers/model_doc/flan-t5)
- [Streamlit Documentation](https://docs.streamlit.io/)

---

## 📝 Sample Document Format

Create text files in the `documents/` folder with content like:

**rag_concepts.txt**
```
Retrieval Augmented Generation (RAG) is a technique that combines retrieval and generation.
RAG enhances language models by providing external knowledge sources.
The RAG pipeline consists of retrieval and generation stages.
Retrieval finds relevant documents for a query.
Generation creates answers based on retrieved documents.
```

---

## 🤝 Contributing

Contributions are welcome! To contribute:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

---

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

---

## 👤 Author

**Guruneela** - [GitHub Profile](https://github.com/guruneela385)

---

## 📞 Support

For issues and questions:
- Open an [issue on GitHub](https://github.com/guruneela385/rag_chatbot/issues)
- Check the Jupyter notebook for detailed walkthrough
- Review code comments in `app.py`

---

## 🙏 Acknowledgments

- Facebook AI Research for FAISS
- Hugging Face for Transformers library
- SBERT maintainers for Sentence Transformers
- Streamlit team for the amazing framework

---

**Happy Chatting with Your Custom Knowledge Base! 🚀**

Last Updated: June 2026
