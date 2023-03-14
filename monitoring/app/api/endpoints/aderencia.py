"""Endpoint para cálculo de aderência."""
from typing import List, Dict
import csv
from fastapi import APIRouter
import pandas as pd
import pickle
from pydantic import BaseModel
from scipy.stats import ks_2samp
import os

router = APIRouter(prefix="/aderencia")


class Item(BaseModel):
    path: str

@router.post("/aderencia")
def adherence(items: List[Item]):
    with open(R"monitoring\model.pkl", "rb") as f:
        model = pickle.load(f)
    
    with open(R"C:\Users\caver\Desktop\Projetos_VS_Code\Neurotech_API\challenge-data-scientist\monitoring\test.csv", newline='') as arquivo:
        dftest = pd.read_csv(arquivo)
    X_test = dftest.drop("TARGET", axis=1)
    y_test = dftest["TARGET"]
    
    result = []
    
    for item in items:
        path = item.path
        path = r"{}".format(path)
        with open(path, newline='') as arquivo2:
            df = pd.read_csv(arquivo2)  

        if "TARGET" in df.columns:
            # Se a coluna "TARGET" estiver presente, assume-se que seja o valor real e calcula-se a aderência com a base de teste
            X = df.drop("TARGET", axis=1)
            y = df["TARGET"]
        else:
            # Caso contrário, usa-se o modelo pré-treinado para gerar a coluna "TARGET"
            X = df
            y = model.predict(X)

        df["score"] = model.predict_proba(X)[:, 1]
        
        ks_stat, p_value = ks_2samp(y_test, y)
        
        result.append({
            "path": path,
            "ks_stat": ks_stat,
            "p_value": p_value
        })

    return result