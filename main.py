from typing import Optional
from fastapi import FastAPI, UploadFile, File, HTTPException, Query
import uvicorn
from ultralytics import YOLO  # type: ignore[attr-defined]
from PIL import Image
import io

app = FastAPI()

modelo = YOLO("yolov8n.pt")


@app.get("/")
def inicio():
    return {"mensagem" : "seja bem vindo!!!"}

@app.post("/detectar")
async def detectar_foto(
    file: UploadFile = File(...),
    esp_base_url: Optional[str] = Query(None, description="URL base do ESP para montar o callback (ex: http://192.168.1.50)"),
):

    nome_arquivo = file.filename
    tipo = file.content_type
    conteudo = await file.read()
    imagem = Image.open(io.BytesIO(conteudo)).convert('RGB')
    resultados = modelo.predict(imagem)
    
    if not tipo or not tipo.startswith("image/"):
        raise HTTPException(400, "Envie apenas imagens")    
    humano = False

    for result in resultados:
        if result.boxes is None:
            continue
        for cls_id in result.boxes.cls:
            nome_classe = modelo.names[int(cls_id)]
            if nome_classe == "person":
                humano = True
                break
        if humano:
            break
    if humano:
        msg = "há humanos nessa imagem"
    else:
        msg = "não há humanos nessa imagem"

    resposta: dict = {
        "mensagem": msg,
        "humanos": humano,
    }
    if esp_base_url:
        base = esp_base_url.rstrip("/")
        resposta["esp_callback"] = f"{base}/set?humano={'1' if humano else '0'}"

    return resposta



# if __name__ == "__main__":
#     uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)