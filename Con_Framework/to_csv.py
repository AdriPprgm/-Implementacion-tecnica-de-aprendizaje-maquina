# Script simple para pasar el dataset de landmines a un archivo CSV para su posterior uso.

import pandas as pd
from ucimlrepo import fetch_ucirepo 
  
# fetch dataset 
land_mines = fetch_ucirepo(id=763) 
  
# data (as pandas dataframes) 
X = land_mines.data.features 
y = land_mines.data.targets 

# Complete dataset
df = pd.DataFrame(X)

# Reemplaza los nombres de las columnas con los nombres correctos
df.columns = ['Voltage', 'Height_sensor', 'Soil_type'] 

df['Mine'] = y

df.to_csv("Con_Framework/data/landmines.csv", index=False)