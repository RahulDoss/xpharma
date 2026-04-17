import os
import json
import requests
from typing import Dict, Any

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from openai import OpenAI

from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors, QED

# =========================
# GEMMA (HF ROUTER)
# =========================
client = OpenAI(
    base_url="https://router.huggingface.co/v1",
    api_key=os.environ["HF_TOKEN"]
)

MODEL_ID = "google/gemma-4-31b-it:novita"

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"]
)


# =========================
# MOLECULE ENGINE (DRUG 3D)
# =========================
class MoleculeEngine:

    @staticmethod
    def build(smiles: str):
        mol = Chem.MolFromSmiles(smiles)
        if not mol:
            return None

        mol = Chem.AddHs(mol)

        AllChem.EmbedMolecule(mol, AllChem.ETKDG())
        AllChem.MMFFOptimizeMolecule(mol)

        pdb = Chem.MolToPDBBlock(mol)

        return {
            "type": "drug",
            "smiles": smiles,
            "pdb": pdb,
            "mw": Descriptors.MolWt(mol),
            "logp": Descriptors.MolLogP(mol),
            "qed": QED.qed(mol)
        }


# =========================
# VACCINE ENGINE (EPITOPE → 3D BACKBONE MODEL)
# =========================
class VaccineEngine:

    @staticmethod
    def peptide_to_pdb(sequence: str):
        """
        Simple alpha-backbone approximation (for visualization only)
        """
        atoms = []
        x = 0.0

        for i, aa in enumerate(sequence):
            atoms.append(f"ATOM  {i+1:4d}  CA  ALA A{i+1:4d}    {x:8.3f}   0.000   0.000")
            x += 1.5

        pdb = "\n".join(atoms)

        return {
            "type": "vaccine",
            "sequence": sequence,
            "pdb": pdb
        }


# =========================
# GEMMA CALL
# =========================
def call_gemma(prompt: str):
    res = client.chat.completions.create(
        model=MODEL_ID,
        messages=[{
            "role": "user",
            "content": [{"type": "text", "text": prompt}]
        }]
    )
    return res.choices[0].message.content


# =========================
# MAIN API
# =========================
@app.post("/api/research")
async def research(payload: Dict[str, str]):

    query = payload["prompt"]

    # =========================
    # 1. INTELLIGENCE LAYER (DECIDE MODE)
    # =========================
    decision = call_gemma(f"""
Classify task:

"{query}"

Return JSON:
{{
  "mode": "drug" or "vaccine",
  "target": "short description",
  "note": "reason"
}}
""")

    try:
        decision = json.loads(decision.replace("```json", "").replace("```", ""))
    except:
        decision = {"mode": "drug", "target": "unknown"}

    results = []

    # =========================
    # 2. DRUG MODE
    # =========================
    if decision["mode"] == "drug":

        raw = call_gemma(f"""
Generate 3 valid SMILES for:
{query}

Return JSON:
{{"candidates": ["", "", ""]}}
""")

        try:
            data = json.loads(raw.replace("```json", "").replace("```", ""))
        except:
            data = {"candidates": ["CCO", "CCN", "c1ccccc1"]}

        for smi in data["candidates"]:
            mol = MoleculeEngine.build(smi)
            if mol:
                results.append(mol)

    # =========================
    # 3. VACCINE MODE
    # =========================
    else:

        epitope = call_gemma(f"""
Generate 3 vaccine epitope peptides for:
{query}

Return JSON:
{{"candidates": ["PEPTIDE1", "PEPTIDE2", "PEPTIDE3"]}}
""")

        try:
            data = json.loads(epitope.replace("```json", "").replace("```", ""))
        except:
            data = {"candidates": ["KLVFFAE", "NATGSKL", "VVLGKTA"]}

        for seq in data["candidates"]:
            pep = VaccineEngine.peptide_to_pdb(seq)
            results.append(pep)

    # =========================
    # 4. AI EXPLANATION
    # =========================
    explanation = call_gemma(f"""
Explain scientific reasoning:

Task: {query}
Mode: {decision["mode"]}

Candidates: {json.dumps(results)}

Give short pharma explanation.
""")

    return {
        "mode": decision["mode"],
        "target": decision.get("target"),
        "results": results,
        "explanation": explanation
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
