import os 
from openai import OpenAI 
from dotenv import load_dotenv 
# Cargar variables de entorno 
load_dotenv() 
# Inicializar cliente con la clave API 
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
# Solicitud al modelo, en este caso para el ejercicio se solicitara una sinopsis de la serie "gurren lagan"
prompt = "Escribe un pequeño resumen de la serie gurren lagann y el mensaje que transmite" 
response = client.chat.completions.create( 
model="gpt-4o-mini", 
messages=[{"role": "user", "content": prompt}], 
temperature=0.6 
) 
# Mostrar respuesta 
print("Respuesta del modelo:") 
print(response.choices[0].message.content) 