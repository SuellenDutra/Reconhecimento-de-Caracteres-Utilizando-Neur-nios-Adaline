import matplotlib.pyplot as plt
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score

# Importamos a nossa mesma base de dados já processada!
from processamento import gerar_e_processar_dados

def main():
    print("Gerando os dados para a versão moderna...")
    # Usamos os mesmos dados: 2 fontes (Arial e Times) para garantir uma comparação justa
    X_dados, y_dados = gerar_e_processar_dados(novo_tamanho=(20, 20))
    
    print("\nIniciando treinamento do Perceptron Multicamadas (MLP)...")
    
    # Criando o modelo moderno:
    # hidden_layer_sizes=(100, 50) significa que temos duas camadas ocultas complexas
    # max_iter=500 é o número máximo de épocas
    modelo_mlp = MLPClassifier(hidden_layer_sizes=(100, 50), max_iter=500, random_state=42)
    
    # O treinamento acontece todo nesta única linha (A biblioteca cuida da matemática pesada)
    modelo_mlp.fit(X_dados, y_dados)
    print("Treinamento concluído!")

    # Avaliação do modelo moderno
    previsoes = modelo_mlp.predict(X_dados)
    acuracia = accuracy_score(y_dados, previsoes)
    print(f"Acurácia do modelo MLP (Modernizado): {acuracia * 100:.2f}%")

    # Gráfico de convergência do MLP (scikit-learn chama de 'loss_curve_')
    plt.figure(figsize=(8, 5))
    plt.plot(modelo_mlp.loss_curve_, color='green', linewidth=2)
    plt.title("Convergência do Erro (Loss) - MLP Moderno")
    plt.xlabel("Épocas (Iterações)")
    plt.ylabel("Erro")
    plt.grid(True)
    plt.show()

if __name__ == "__main__":
    main()