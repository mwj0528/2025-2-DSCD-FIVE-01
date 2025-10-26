# peek_hscode_fixed.py
import chromadb, json

PATH = r"../chroma_db"
COL  = "hscode_collection"

client = chromadb.PersistentClient(path=PATH)
col = client.get_collection(COL)

# ✅ include에서 'ids' 제거 (get은 ids를 자동 반환)
res = col.get(include=["metadatas", "documents"], limit=5)

print(f"[{COL}] 샘플 {len(res.get('ids', []))}건")
for i in range(len(res.get("ids", []))):
    print("="*60)
    print("ID:", res["ids"][i])
    meta = (res["metadatas"][i] or {})
    print("META keys:", list(meta.keys()))
    print("META:", json.dumps(meta, ensure_ascii=False))
    doc = (res["documents"][i] or "")
    print("DOC:", (doc[:300] + ("..." if len(doc) > 300 else "")))

