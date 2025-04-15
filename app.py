#!/usr/bin/env python3
import argparse
import os
from simple_rag.cli import main as simple_rag_main
from contextual_rag.cli import main as contextual_rag_main

def main():
    """Main entry point for the application"""
    parser = argparse.ArgumentParser(description='RAG System with Cohere')
    subparsers = parser.add_subparsers(dest='command', help='RAG System to use')
    
    # Simple RAG subparser
    simple_parser = subparsers.add_parser('simple', help='Use simple RAG system')
    simple_parser.add_argument('--pdf_path', type=str, required=True, 
                        help='Path to PDF file or directory containing PDF files')
    simple_parser.add_argument('--query', type=str, help='Question to ask')
    simple_parser.add_argument('--interactive', action='store_true', 
                        help='Run in interactive mode')
    
    # Contextual RAG subparser
    contextual_parser = subparsers.add_parser('contextual', help='Use contextual RAG system')
    contextual_parser.add_argument('--pdf_path', type=str, required=True, 
                        help='Path to PDF file or directory containing PDF files')
    contextual_parser.add_argument('--query', type=str, help='Question to ask')
    contextual_parser.add_argument('--interactive', action='store_true', 
                        help='Run in interactive mode')
    
    args = parser.parse_args()
    
    if args.command == 'simple':
        # Call simple RAG CLI with the parsed arguments
        os.environ['PDF_PATH'] = args.pdf_path
        if args.query:
            os.environ['QUERY'] = args.query
        os.environ['INTERACTIVE'] = str(args.interactive)
        simple_rag_main()
    
    elif args.command == 'contextual':
        # Call contextual RAG CLI with the parsed arguments
        os.environ['PDF_PATH'] = args.pdf_path
        if args.query:
            os.environ['QUERY'] = args.query
        os.environ['INTERACTIVE'] = str(args.interactive)
        contextual_rag_main()
    
    else:
        parser.print_help()

if __name__ == "__main__":
    main()