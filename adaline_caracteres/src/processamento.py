import os
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

def gerar_e_processar_dados(novo_tamanho=(20, 20)):
    alfabeto = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    caminho_dados = os.path.join(os.path.dirname(__file__), '..', 'dados', 'treino')
    os.makedirs(caminho_dados, exist_ok=True)
    
    X = []
    y = []
    nomes_fontes = ['arial.ttf', 'times.ttf']
    
    for indice, letra in enumerate(alfabeto):
        for nome_fonte in nomes_fontes:
            
            #cria a imagem
            image = Image.new('RGB', (50, 50), color=(255, 255, 255))
            draw = ImageDraw.Draw(image)
            
            try:
                fonte = ImageFont.truetype(nome_fonte, 40) 
            except IOError:
                fonte = ImageFont.load_default()
            
            draw.text((10, 0), letra, fill=(0, 0, 0), font=fonte)
            
            nome_limpo_fonte = nome_fonte.split('.')[0]
            caminho_arquivo = os.path.join(caminho_dados, f'{letra}_{nome_limpo_fonte}.png')
            image.save(caminho_arquivo)
            
            #converte a imagem do PIL para um array
            image_np = np.array(image)

            #converte de RGB para BGR
            image_bgr = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)

            #converte para escala de cinza (preto e branco)
            gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)

            #aplica o limiar (fundo vira 0, a letra vira 1)
            _, binaria = cv2.threshold(gray, 127, 1, cv2.THRESH_BINARY_INV)
            
            # tamanho padrão da rede neural (20x20)
            binaria_reduzida = cv2.resize(binaria, novo_tamanho)
            
            X.append(binaria_reduzida.flatten())#transforma em números
            y.append(indice) 
        
    return np.array(X), np.array(y)

def processar_imagem_unica(caminho_imagem, novo_tamanho=(20, 20)):
    """
    Lê uma única imagem do computador, aplica os mesmos filtros do treino
    e devolve o vetor de 400 posições para a interface.
    """
    # Lê a imagem direto do disco
    image_bgr = cv2.imread(caminho_imagem)
    
    # Converte para escala de cinza
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    
    # Aplica o limiar (fundo vira 0, letra vira 1)
    _, binaria = cv2.threshold(gray, 127, 1, cv2.THRESH_BINARY_INV)
    
    # Redimensiona para 20x20
    binaria_reduzida = cv2.resize(binaria, novo_tamanho)
    
    # Achata para virar o vetor de 400 números e devolve
    return binaria_reduzida.flatten()