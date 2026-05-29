import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk

import processamento
import treino 

print("\n[INTERFACE] Iniciando o pipeline de treinamento...") #roda o treino.py
cerebro_da_ia = treino.main()

alfabeto = ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L", "M", 
            "N", "O", "P", "Q", "R", "S", "T", "U", "V", "W", "X", "Y", "Z"]

def classificar_imagem_na_ia(caminho_da_imagem):
    vetor_numeros = processamento.processar_imagem_unica(caminho_da_imagem) #imagem escolhida na interface
    
    indice_vencedor = cerebro_da_ia.prever(vetor_numeros) #previsão de qual letra é
    
    letra_reconhecida = alfabeto[indice_vencedor] #traduz o número
    
    return letra_reconhecida

#lógica dos botões
def escolher_imagem(): #abre a janela do arquivo
    caminho_arquivo = filedialog.askopenfilename(
        title="Escolha a imagem da letra",
        filetypes=[("Imagens", "*.png *.jpg *.jpeg *.bmp")]
    )
    
    if caminho_arquivo: # Mostra a imagem na tela
        img = Image.open(caminho_arquivo)
        img = img.resize((150, 150))
        img_tk = ImageTk.PhotoImage(img)
        label_imagem.config(image=img_tk)
        label_imagem.image = img_tk
        
        texto_resultado.config(text="Processando...", fg="blue") #manda a imagem e atualiza o texto
        janela.update()
        
        letra = classificar_imagem_na_ia(caminho_arquivo) #chama a função

        texto_resultado.config(text=f"LETRA RECONHECIDA: {letra}", fg="green") #resultado

#cria a janela principal
janela = tk.Tk()
janela.title("Reconhecimento de Caracteres - Adaline")
janela.geometry("500x500")

titulo = tk.Label(janela, text="Sistema de Visão Computacional", font=("Arial", 16, "bold"))
titulo.pack(pady=20)

#botão para escolher a imagem
botao_escolher = tk.Button(janela, text="Escolher Imagem", command=escolher_imagem, font=("Arial", 12), bg="#e0e0e0")
botao_escolher.pack(pady=10)

#espaço onde a imagem vai aparecer
label_imagem = tk.Label(janela)
label_imagem.pack(pady=20)

#texto
texto_resultado = tk.Label(janela, text="Aguardando imagem...", font=("Arial", 18, "bold"), fg="gray")
texto_resultado.pack(pady=20)

#abre a janela
janela.mainloop()