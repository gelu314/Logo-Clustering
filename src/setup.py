import pandas as pd
from PIL import Image
import os
import requests
from io import BytesIO

#run this file to download the logos from the domains in the parquet file

def setup():

    file_path = 'data/logos.snappy.parquet' #this is where the logo names are stored
    destination = 'data/logos'
    img_size = (128, 128)

    df = pd.read_parquet(file_path)

    for i in range(len(df)):
        domain = df['domain'].iloc[i]
        domain = domain.replace('www.', '').replace('http://', '').replace('https://', '').split('/')[0]

        print(f"Domain {i+1}/{len(df)}: {domain}")

        logo_url = f"https://logo.clearbit.com/{domain}?size=128"

        try:
            response = requests.get(logo_url, timeout=10)
            response.raise_for_status()

            image = Image.open(BytesIO(response.content))
            image = image.resize(img_size, Image.Resampling.LANCZOS)

            file_name = f"{domain.replace('.', '_')}.png"
            file_path = os.path.join(destination, file_name)
            image.save(file_path,   format='PNG')
            print(f"   -> Success: Saved standardized logo to {file_path}")

        except requests.exceptions.RequestException as e:
            # If Clearbit fails (404), you can try a fallback URL (e.g., Google)
            print(f"   -> Clearbit failed for {domain}. Trying Google fallback...")
            
            # 2. FALLBACK URL (Google)
            fallback_url = f"https://www.google.com/s2/favicons?domain={domain}&sz=128"
            
            try:
                response = requests.get(fallback_url, timeout=10)
                response.raise_for_status()
                
                image = Image.open(BytesIO(response.content))
                image = image.resize(img_size, Image.Resampling.LANCZOS)

                file_name = f"{domain.replace('.', '_')}.png"
                file_path = os.path.join(destination, file_name)
                image.save(file_path, format='PNG')
                print(f"   -> Success: Saved fallback logo to {file_path}")
                
            except requests.exceptions.RequestException as e_fallback:
                print(f"   -> Both services failed for {domain}. Skipping.")


