import time
from typing import List, Dict, Optional
from Bio import Entrez, Medline


class PubMedRetriever:
    """Retrieves abstracts from PubMed using NCBI E-utilities (free, no key required)."""

    def __init__(self, email: str = "medreason.rag@example.com", api_key: Optional[str] = None):
        Entrez.email = email
        if api_key:
            Entrez.api_key = api_key
        # Without API key: 3 req/sec max; with key: 10 req/sec
        self._delay = 0.35 if api_key else 0.4

    def search(self, query: str, max_results: int = 20) -> List[Dict]:
        """Search PubMed and return structured paper records."""
        try:
            handle = Entrez.esearch(
                db="pubmed",
                term=query,
                retmax=max_results,
                sort="relevance",
                usehistory="y",
            )
            record = Entrez.read(handle)
            handle.close()

            pmids = record.get("IdList", [])
            if not pmids:
                return []

            time.sleep(self._delay)

            handle = Entrez.efetch(
                db="pubmed",
                id=",".join(pmids),
                rettype="medline",
                retmode="text",
            )
            records = list(Medline.parse(handle))
            handle.close()

            papers = []
            for r in records:
                abstract = r.get("AB", "")
                if not abstract:
                    continue
                papers.append({
                    "pmid": r.get("PMID", ""),
                    "title": r.get("TI", "No title"),
                    "abstract": abstract,
                    "authors": r.get("AU", []),
                    "journal": r.get("TA", ""),
                    "year": (r.get("DP", "") or "")[:4],
                    "mesh_terms": r.get("MH", []),
                    "url": f"https://pubmed.ncbi.nlm.nih.gov/{r.get('PMID', '')}/",
                })

            return papers

        except Exception as e:
            print(f"[PubMed] Search error for query '{query[:60]}': {e}")
            return []

    def multi_query_search(self, queries: List[str], max_per_query: int = 10) -> List[Dict]:
        """Run multiple queries, deduplicate by PMID, return merged list."""
        seen: Dict[str, Dict] = {}
        for query in queries[:3]:  # cap at 3 to stay within rate limits
            for paper in self.search(query, max_per_query):
                pmid = paper["pmid"]
                if pmid and pmid not in seen:
                    seen[pmid] = paper
            time.sleep(self._delay)
        return list(seen.values())
