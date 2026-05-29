import numpy as np

class AdalineOneVsAll:
    def __init__(self, tamanho_entrada, num_classes=26, taxa_aprendizado=0.01):
        self.taxa_aprendizado = taxa_aprendizado
        self.num_classes = num_classes
        self.pesos = np.random.randn(num_classes, tamanho_entrada) * 0.01
        self.bias = np.random.randn(num_classes) * 0.01
        self.historico_erro = []

    def treinar(self, X, y, epocas=100):
        for epoca in range(epocas):
            erro_total_epoca = 0
            
            for i in range(len(X)):
                xi = X[i]
                t = np.full(self.num_classes, -1.0)
                t[y[i]] = 1.0 
                
                saida_linear = np.dot(self.pesos, xi) + self.bias
                erro = t - saida_linear
                erro_total_epoca += 0.5 * np.sum(erro**2)
                
                for c in range(self.num_classes):
                    self.pesos[c] += self.taxa_aprendizado * erro[c] * xi
                    self.bias[c] += self.taxa_aprendizado * erro[c]
            
            mse = erro_total_epoca / (len(X) * self.num_classes)
            self.historico_erro.append(mse)
            print(f"Época {epoca + 1}/{epocas} - MSE: {mse:.4f}")

    def prever(self, X):
        saidas_lineares = np.dot(X, self.pesos.T) + self.bias
        
        # Agora ele sabe lidar com uma imagem só (da interface) ou várias (do treino)!
        if saidas_lineares.ndim == 1:
            return np.argmax(saidas_lineares)
        else:
            return np.argmax(saidas_lineares, axis=1)