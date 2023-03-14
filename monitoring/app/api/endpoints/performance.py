"""Endpoint para cálculo de Performance."""
import json
from typing import List,Dict
from fastapi import APIRouter
import numpy as np
from pydantic import BaseModel
from datetime import datetime
from collections import defaultdict
import pandas as pd
import pickle
from sklearn.metrics import roc_auc_score




router = APIRouter(prefix="/performance")


class InputData(BaseModel):
    VAR2: str
    IDADE: int
    VAR5: str
    VAR6: float
    VAR7: float
    VAR8: str
    VAR9: str
    VAR10: str
    VAR11: float
    VAR12: float
    VAR14: float
    VAR15: float
    VAR16: float
    VAR18: float
    VAR19: float
    VAR22: float
    VAR24: float
    VAR25: float
    VAR32: str
    VAR39: float
    VAR40: float
    VAR41: float
    VAR42: float
    VAR47: float
    VAR49: str
    VAR50: str
    VAR51: str
    VAR52: str
    VAR53: str
    VAR54: str
    VAR55: str
    VAR56: str
    VAR57: str
    VAR58: str
    VAR59: str
    VAR60: str
    VAR61: str
    VAR62: str
    VAR63: str
    VAR64: str
    VAR65: str
    VAR66: str
    VAR67: str
    VAR68: str
    VAR69: str
    VAR70: str
    VAR71: str
    VAR72: str
    VAR73: str
    VAR74: str
    VAR75: str
    VAR76: str
    VAR77: str
    VAR78: str
    VAR79: str
    VAR80: str
    VAR81: str
    VAR82: str
    VAR83: str
    VAR84: str
    VAR85: str
    VAR86: str
    VAR87: str
    VAR88: str
    VAR89: str
    VAR90: str
    VAR91: str
    VAR92: str
    VAR93: str
    VAR94: str
    VAR95: str
    VAR96: str
    VAR97: str
    VAR98: str
    VAR99: str
    VAR100: str
    VAR101: str
    VAR102: str
    VAR103: str
    VAR104: str
    VAR105: str
    VAR106: str
    VAR107: str
    VAR108: str
    VAR109: str
    VAR110: str
    VAR111: str
    VAR112: str
    VAR113: str
    VAR114: str
    VAR115: str
    VAR116: str
    VAR117: str
    VAR118: str
    VAR119: str
    VAR120: str
    VAR121: str
    VAR122: str
    VAR123: str
    VAR124: str
    VAR125: str
    VAR126: str
    VAR127: str
    VAR128: str
    VAR129: str
    VAR130: str
    VAR131: str
    VAR132: str
    VAR133: str
    VAR134: str
    VAR135: str
    VAR136: str
    VAR137: str
    VAR138: str
    VAR139: str
    VAR140: str
    VAR141: float
    VAR142: str
    REF_DATE: str
    TARGET: int



def sum_counts_by_month(counts_list):
    totals = defaultdict(int)
    for counts in counts_list:
        for month, count in counts.items():
            totals[month] += count
    return dict(totals)


def replace_nones_with_nans(data: Dict) -> Dict:
    """
    Replace None values with 0 in a dictionary
    """
    return {k: np.nan if v is None else v for k, v in data.items()}




@router.post("/performance-volumetry")
def volumetry(data_list: List[Dict]):


    #Pegando quantidade de registros por datas:
    counts_list = []
    for data in data_list:
        # replace None values with NaN in the data dictionary
        data = replace_nones_with_nans(data)
        counts = defaultdict(int)
        ref_date = datetime.fromisoformat(data['REF_DATE']).date()
        for month in range(1, 13):
            date = datetime(ref_date.year, month, 1).date()
            counts[date.strftime('%Y-%m')] = 0
        counts[ref_date.strftime('%Y-%m')] = len(data) - 1
        counts_list.append(dict(counts))
    all_bodies = sum_counts_by_month(counts_list)

    #Pegando a performance usando o valor da área sob a curva ROC
    #Fazer um loop que itera para todos os bodies, cada body e uma linha, formar um pandas data frame com cada linha sendo um body
    data = json.loads(json.dumps(data_list))
    df = pd.json_normalize(data)
    df = df.fillna(value=np.nan)

    with open(R'C:\Users\caver\Desktop\Projetos_VS_Code\Neurotech_API\challenge-data-scientist\monitoring\model.pkl', 'rb') as f:
        modelo = pickle.load(f)
    
    y_pred = modelo.predict_proba(df.drop('TARGET', axis=1))[:, 1]

    roc_auc = roc_auc_score(df['TARGET'], y_pred)
    
    return all_bodies, ("valor da área sob a curva ROC: "), roc_auc