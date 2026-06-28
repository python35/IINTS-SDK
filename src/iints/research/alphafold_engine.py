import urllib.request
import json
from typing import Dict, Any

class AlphaFoldGenomicsEngine:
    """
    Bridges deep 3D structural AI (AlphaFold) to physiological equations.
    """

    @staticmethod
    def evaluate_plddt_impact(uniprot_id: str, residue_index: int) -> Dict[str, Any]:
        """
        Fetches AlphaFold structure, finds the pLDDT for the residue,
        and mathematically translates it into a molecular_affinity_scalar.
        """
        api_url = f"https://alphafold.ebi.ac.uk/api/prediction/{uniprot_id}"
        
        try:
            req = urllib.request.Request(api_url, headers={'User-Agent': 'IINTS-AF-SDK/1.0'})
            with urllib.request.urlopen(req) as response:
                if response.status != 200:
                    return {"error": f"Failed to query AlphaFold API for {uniprot_id} (Status {response.status})"}
                data = json.loads(response.read().decode())
        except Exception as e:
            return {"error": f"AlphaFold API request failed: {e}"}

        if not data:
            return {"error": f"No AlphaFold prediction found for {uniprot_id}"}
            
        # Find the correct fragment for large proteins
        target_fragment = None
        for frag in data:
            start = frag.get("uniprotStart", 1)
            end = frag.get("uniprotEnd", 999999)
            if start <= residue_index <= end:
                target_fragment = frag
                break
                
        if not target_fragment:
            return {"error": f"Residue {residue_index} is out of bounds for {uniprot_id}."}
            
        pdb_url = target_fragment.get("pdbUrl")
        if not pdb_url:
            return {"error": "PDB file not available in AlphaFold DB for this protein."}
            
        try:
            pdb_req = urllib.request.Request(pdb_url, headers={'User-Agent': 'IINTS-AF-SDK/1.0'})
            with urllib.request.urlopen(pdb_req) as response:
                pdb_text = response.read().decode()
        except Exception as e:
            return {"error": f"Failed to download PDB file: {e}"}
            
        # Parse PDB text to find pLDDT (B-factor is in columns 61-66)
        plddt_values = []
        for line in pdb_text.split("\n"):
            if line.startswith("ATOM  "):
                try:
                    res_num = int(line[22:26].strip())
                    if res_num == residue_index:
                        b_factor = float(line[60:66].strip())
                        plddt_values.append(b_factor)
                except ValueError:
                    pass
                    
        if not plddt_values:
            return {"error": f"Residue {residue_index} not found in AlphaFold structure."}
            
        avg_plddt = sum(plddt_values) / len(plddt_values)
        
        # Mathematical translation
        if avg_plddt >= 90:
            scalar = 0.15
            conclusion = "Highly structured core domain. Mutation is likely catastrophic."
        elif avg_plddt >= 70:
            scalar = 0.40
            conclusion = "Structured domain. Mutation likely impairs function."
        elif avg_plddt >= 50:
            scalar = 0.75
            conclusion = "Moderate flexibility. Mutation partially tolerated."
        else:
            scalar = 0.95
            conclusion = "Intrinsically disordered/flexible region. Mutation is highly tolerated."
            
        return {
            "uniprot_id": uniprot_id,
            "residue_index": residue_index,
            "plddt": round(avg_plddt, 2),
            "scalar": scalar,
            "conclusion": conclusion,
            "pdb_url": pdb_url
        }
