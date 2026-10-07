import json
import urllib.request
import numpy as np

# 1. Load the Ensembl IDs from your npy file
genes = np.load("BRCA_Omni_Genes.npy", allow_pickle=True)

# 2. Strip version numbers (e.g., ENSG00000145982.10 -> ENSG00000145982)
ensg_map = {str(g).split('.')[0]: str(g) for g in genes}
clean_ids = list(ensg_map.keys())

# 3. Query the MyGene.info API
url = "https://mygene.info/v3/query"
payload = json.dumps({"q": clean_ids, "scopes": "ensembl.gene", "fields": "symbol,name"}).encode('utf-8')
req = urllib.request.Request(url, data=payload, headers={'Content-Type': 'application/json'})

print(f"{'Ensembl ID':<22} | {'Gene Symbol':<12} | {'Gene Description'}")
print("=" * 80)

mapped_results = []

try:
    with urllib.request.urlopen(req) as response:
        results = json.loads(response.read().decode('utf-8'))
        
    for item in results:
        clean_id = item.get('query', '')
        full_ensg = ensg_map.get(clean_id, clean_id)
        symbol = item.get('symbol', 'N/A')
        name = item.get('name', 'N/A')
        
        mapped_results.append((full_ensg, symbol, name))
        print(f"{full_ensg:<22} | {symbol:<12} | {name}")

except Exception as e:
    print(f"[!] Error mapping genes: {e}")

# 4. Save mapped output as a CSV for easy inclusion in your manuscript
with open("BRCA_50_Gene_Panel_Mapped.csv", "w") as f:
    f.write("Ensembl_ID,Gene_Symbol,Gene_Description\n")
    for full_ensg, symbol, name in mapped_results:
        clean_name = name.replace(',', ';') # avoid breaking CSV formatting
        f.write(f"{full_ensg},{symbol},{clean_name}\n")

print("\n[OK] Mapped results saved to 'BRCA_50_Gene_Panel_Mapped.csv'")