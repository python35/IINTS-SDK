import urllib.request
import json
from typing import Dict, Any

class AlphaFoldGenomicsEngine:
    """
    Retrieves residue-level AlphaFold confidence for structural context.

    AlphaFold pLDDT is a confidence score for the predicted local structure. It
    is not a pathogenicity score and must not be converted into a physiological
    effect size, binding-affinity loss, or clinical classification.
    """

    @staticmethod
    def evaluate_plddt_impact(uniprot_id: str, residue_index: int) -> Dict[str, Any]:
        """
        Fetch the local pLDDT score without inferring a mutation effect.

        The historical method name is retained for API compatibility. Callers
        must use a separately sourced functional effect size before changing a
        patient-model parameter.
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
        
        # pLDDT bands describe model confidence only. They do not establish
        # conservation, pathogenicity, receptor function, or binding affinity.
        if avg_plddt >= 90:
            confidence_band = "very_high"
        elif avg_plddt >= 70:
            confidence_band = "confident"
        elif avg_plddt >= 50:
            confidence_band = "low"
        else:
            confidence_band = "very_low"

        conclusion = (
            "Local AlphaFold prediction confidence only; no mutation severity, "
            "pathogenicity, or functional effect is inferred."
        )
            
        return {
            "uniprot_id": uniprot_id,
            "residue_index": residue_index,
            "plddt": round(avg_plddt, 2),
            "confidence_band": confidence_band,
            "supports_functional_inference": False,
            "conclusion": conclusion,
            "pdb_url": pdb_url
        }
