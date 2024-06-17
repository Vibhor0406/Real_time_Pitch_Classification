import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List
from utils import get_pitch_mix, get_available_pitcher_ids
import joblib

app = FastAPI()
pitcher_ids = get_available_pitcher_ids()


# define model input data structure for a pitch
class Pitch(BaseModel):
    release_speed: List[float]
    release_spin_rate: List[float]
    pfx_x: List[float]
    pfx_z: List[float]
    stand: List[str]


# define model input data structure for a pitcher
class Pitcher(BaseModel):
    id: int


@app.get("/")
def root():
    return {"Hello": "World!"}


# allow users to get a pitch mix
@app.post("/mix")
def pitch_mix(pitcher: Pitcher):
    if pitcher.id not in pitcher_ids:
        raise HTTPException(status_code=404, detail="Pitcher not available.")
    mix = get_pitch_mix(pitcher.id)
    return mix.to_json()


# allow users to get a pitch type prediction
@app.post("/predict")
def classify_pitch(pitcher: Pitcher, pitch: Pitch):
    if pitcher.id not in pitcher_ids:
        raise HTTPException(status_code=404, detail="Pitcher not available.")
    model = joblib.load(f"models/{pitcher.id}.joblib")
    df = pd.DataFrame(pitch.model_dump())
    prediction = model.predict(df)
    if prediction == ['CH']:
        return "Changeup"
    elif prediction == ['CU']:
        return "Curveball"
    elif prediction == ['FC']:
        return "Cutter"
    elif prediction == ['EP']:
        return "Eephus"
    elif prediction == ['FO']:
        return "Forkball"
    elif prediction == ['FF']:
        return "Four-Seam Fastball"
    elif prediction == ['KN']:
        return "Knuckleball"
    elif prediction == ['KC']:
        return "Knuckle-curve"
    elif prediction == ['SC']:
        return "Screwball"
    elif prediction == ['SI']:
        return "Sinker"
    elif prediction == ['SL']:
        return "Slider"
    elif prediction == ['SV']:
        return "Slurve"
    elif prediction == ['FS']:
        return "Splitter"
    elif prediction == ['ST']:
        return "Sweeper"
    else:
        return "Unknown pitch type"
    
