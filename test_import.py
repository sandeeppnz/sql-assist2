try:
    from vanna.chromadb import ChromaDB_VectorStore
    print("SUCCESS: vanna.chromadb import works")
except ImportError as e:
    print(f"ERROR with vanna.chromadb: {e}")
    try:
        from vanna.integrations.local import ChromaDB_VectorStore
        print("SUCCESS: vanna.integrations.local import works")
    except ImportError as e2:
        print(f"ERROR with vanna.integrations.local: {e2}")
        try:
            import vanna.local
            print("Checking vanna.local...")
            print(dir(vanna.local))
        except Exception as e3:
            print(f"ERROR with vanna.local: {e3}")

