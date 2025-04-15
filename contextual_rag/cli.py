import argparse
import os
from contextual_rag.modules.embedding import init_embeddings, init_llm
from contextual_rag.modules.pdf_loader import ContextualPDFProcessor
from contextual_rag.modules.qa_chain import ContextualQAChain

def main():
    parser = argparse.ArgumentParser(description='Contextual RAG System with Cohere')
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
    pdf_processor = ContextualPDFProcessor(embeddings, llm)
    
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
    qa_chain = ContextualQAChain(pdf_processor.vector_store, llm)
    
    if args.interactive:
        # Interactive mode
        print("\nEntering interactive mode. Type 'exit' to quit.")
        conversation_history = []
        
        while True:
            query = input("\nYour question: ")
            if query.lower() == 'exit':
                break
            
            answer = qa_chain.generate_answer(query, conversation_history)
            print(f"\nAnswer: {answer}")
            
            # Update conversation history
            conversation_history.append({"role": "user", "content": query})
            conversation_history.append({"role": "assistant", "content": answer})
    
    elif args.query:
        # Single question mode
        answer = qa_chain.generate_answer(args.query)
        print(f"\nQuestion: {args.query}")
        print(f"Answer: {answer}")
    
    else:
        print("Please provide a query using --query or use --interactive mode")

if __name__ == "__main__":
    main()