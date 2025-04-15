import argparse
import os
from simple_rag.modules.embedding import init_embeddings, init_llm
from simple_rag.modules.pdf_loader import PDFProcessor
from simple_rag.modules.qa_chain import QAChain

def main():
    parser = argparse.ArgumentParser(description='Simple RAG System with Cohere')
    parser.add_argument('--pdf_path', type=str, required=True, 
                        help='Path to PDF file or directory containing PDF files')
    parser.add_argument('--query', type=str, help='Question to ask')
    parser.add_argument('--interactive', action='store_true', 
                        help='Run in interactive mode')
    
    args = parser.parse_args()
    
    # Initialize embeddings and LLM
    embeddings = init_embeddings()
    llm = init_llm()
    
    # Initialize PDF processor
    pdf_processor = PDFProcessor(embeddings)
    
    # Process PDF(s)
    pdf_paths = []
    if os.path.isdir(args.pdf_path):
        for file in os.listdir(args.pdf_path):
            if file.endswith('.pdf'):
                pdf_paths.append(os.path.join(args.pdf_path, file))
    else:
        pdf_paths = [args.pdf_path]
    
    for pdf_path in pdf_paths:
        print(f"Processing: {pdf_path}")
        pdf_processor.load_and_process(pdf_path)
    
    # Initialize QA chain
    qa_chain = QAChain(pdf_processor.vector_store, llm)
    
    if args.interactive:
        # Interactive mode
        print("\nEntering interactive mode. Type 'exit' to quit.")
        while True:
            query = input("\nYour question: ")
            if query.lower() == 'exit':
                break
            
            answer = qa_chain.generate_answer(query)
            print(f"\nAnswer: {answer}")
    
    elif args.query:
        # Single question mode
        answer = qa_chain.generate_answer(args.query)
        print(f"\nQuestion: {args.query}")
        print(f"Answer: {answer}")
    
    else:
        print("Please provide a query using --query or use --interactive mode")

if __name__ == "__main__":
    main()