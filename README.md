## TODO
1. Chunking in Base parser should be advanced.
2. Accept Json objects and Directly convert them to Graph.
3. document_parser.py contain a list of results.
4. Learn how pytesseract works.
5. Once the type of P&IDs are finalized change _extract_entities_heurisitc 
   from vision_parser.
6. Make vision_client model structured for now Later Try to use Yolo pipeline. *********
7. Change _extract_topological_relationships in vision_parser it extract both entities, 
   relatioins and little summary (chunk it too and make embeddings) with structured vision_client. ***
8. In semantic_parser try to use custom_model instead of pyteseract (Can be used in VisionParer too).
9. Make a own chunking method for spreadsheets to make it row aware.
10. Check spreadsheets, archive parser again. ***********
11. Learn rapidfuzz.
12. If possible in sql_tools/_fetch_neo4j_procedures_sync change exact_match.
13. While building Graph strict enforce it to have schema required for compliance agent
    and rca agent. *********
15. In intelligenc_router add other router or service as soon as user request is finished it should be sent to user.
16. RCA should use real TIME data too. ***** (etl,timescale db)

18. After user queries something the evidence should be clickable.(should return links of soure files to frontend).

19. Learn about ayncio and .gather all imp methods.

20. Update the cyper query in knowledge copiolt.

21. In knowledge copilot missing retry for grade_context and on errors it is directly going to next node.

22. CRITICAl in file classifier data is not writtten int neo4j in last _write_to_db method.

23. CRITICAL for all parsers parsing garaph data make sure to resolve them using entity parser.

24. In spread sheet parser use custom chunker for making rows intact.

25. CRITICAL Enforce req parsers to create graph data satisfying other agens.


## Check Complete
1. Emails
2. Visiion Parser
3. Semantic parser
4. Digital_text paraser
5. ingestion_router
6. file classifier
7. rca_agent_router and related.
8. copiolt. 


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
7. digital_text_parser can parser typed pdfs.
8. Document parser order is vision-> semantic->digital_text_parser.
9. Compilance agent assumes graph look like this 
   (Regulation)
         │
         │ HAS_CLAUSE
         ▼
      (Clause)
         │
         │ APPLIES_TO
         ▼
   (AssetType)
10. RCA agent assumes graph look like this
   AssetType
      │
   HAS_DOCUMENT
      ▼
   Document
      │
   CONTAINS_PROCEDURE
      ▼
   Procedure


11. knowledge_copilot takes best case 5 llm api calls at worst 11 api calls.

12. Using vector search in neo4j.

## Final optimizations
1. Interpret what happens on scale write its impact and
   how application takes care of it. (business impact).

2. while inserting data into graph try to push embeddings in formate.