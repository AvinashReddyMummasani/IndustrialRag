## TODO
1. Chunking in Base parser should be advanced.
2. Accept Json objects and Directly convert them to Graph.
3. document_parser.py contain a list of results.
4. Learn how pytesseract works.
5. Once the type of P&IDs are finalized change _extract_entities_heurisitc 
   from vision_parser.
6. Make vision_client model structured for now Later Try to use Yolo pipeline. ***
7. Change _extract_topological_relationships in vision_parser it extract both entities, 
   relatioins and little summary (chunk it too and make embeddings) with structured vision_client. ***
8. In semantic_parser try to use custom_model instead of pyteseract (Can be used in VisionParer too).
9. 

## Check Complete
1. Emails
2. Visiion Parser



## Notes

1. fitz lib can proces pdfs and return a Document.
   eg:
    Document
    │
    ├── Page 0 (Page object)
    │     ├── Text
    │     ├── Images
    │     ├── Tables
    │     └── Annotations
    ├── Page 1
    └── Page 2
2. File ingestion happens in Background.
3. Pdfs can handle p&ids, scanned images or text, text.
4. Vision parser accepts imgs, scanned images.
5. Eamil parser accepts emails, messages.
6. Semantic parser handles scanned text.