import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score

from processamento import gerar_e_processar_dados
from adaline import AdalineOneVsAll

def imprimir_letra_no_terminal(vetor, tamanho=(20, 20)):
    print("-" * 65)
    matriz = vetor.reshape(tamanho)
    
    for linha in matriz:
        linha_texto = ""
        for pixel in linha:
            if pixel == 1:
                linha_texto += "  1"  # Espaço para manter alinhado
            else:
                linha_texto += " -1"  # fundo como -1
        print(linha_texto)
    print("-" * 65)

def main():
    print("Gerando e processando imagens...")
    X_dados, y_dados = gerar_e_processar_dados(novo_tamanho=(20, 20))
    tamanho_entrada = X_dados.shape[1]
    
    print("\nLetra 'A' lida pela IA:")
    imprimir_letra_no_terminal(X_dados[8])
    input("\nPressione ENTER para iniciar o treinamento e ver as épocas...")
    
    print("\nIniciando treinamento do Adaline...")
    modelo_adaline = AdalineOneVsAll(tamanho_entrada=tamanho_entrada, taxa_aprendizado=0.01)
    modelo_adaline.treinar(X_dados, y_dados, epocas=50)
    print("Treinamento concluído.\n")

    # Avaliação
    previsoes = modelo_adaline.prever(X_dados)
    acuracia = accuracy_score(y_dados, previsoes)
    print(f"Acurácia do modelo: {acuracia * 100:.2f}%")
    
    # Gráfico
    plt.figure(figsize=(8, 5))
    plt.plot(modelo_adaline.historico_erro, color='blue', linewidth=2)
    plt.title("Convergência do Erro Quadrático Médio (MSE) - Adaline")
    plt.xlabel("Épocas")
    plt.ylabel("MSE")
    plt.grid(True)
    plt.show()
    
    return modelo_adaline

if __name__ == "__main__":
    main()

# python src/treino.py