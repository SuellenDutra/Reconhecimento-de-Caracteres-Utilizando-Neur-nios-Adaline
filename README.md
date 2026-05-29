# Reconhecimento de Caracteres com Rede Neural Adaline 🧠🔤

## 📖 Sobre o Projeto
Este projeto implementa uma rede neural **Adaline (Adaptive Linear Neuron)** construída do zero para o reconhecimento de caracteres alfabéticos (A-Z) a partir de imagens. O sistema utiliza a estratégia de classificação multiclasse **One-vs-All** e conta com uma interface gráfica intuitiva para a realização de testes em tempo real.

## ✨ Funcionalidades
* **Treinamento "From Scratch":** Implementação matemática pura da Regra Delta, sem o uso de frameworks de alto nível (como TensorFlow ou PyTorch) para o núcleo de aprendizagem.
* **Processamento de Visão Computacional:** Binarização, redimensionamento e extração de características de imagens utilizando OpenCV.
* **Estratégia One-vs-All:** Treinamento simultâneo de 26 neurônios independentes, onde cada um atua como um especialista para uma letra específica.
* **Análise de Convergência:** Geração automática do gráfico de Erro Quadrático Médio (MSE) para comprovar o aprendizado do modelo matemático.
* **Interface Gráfica (GUI):** Tela interativa construída com Tkinter para que o usuário possa testar o modelo treinado com novas imagens.

## 📂 Estrutura do Projeto
O sistema foi desenhado sob uma arquitetura modular, dividindo claramente as responsabilidades:

* `processamento.py`: O tradutor de dados. Transforma imagens (`.png`/`.jpg`) em vetores matemáticos (arrays de 400 posições contendo `-1` e `1`) utilizando conversão para escala de cinza e limiarização (threshold).
* `adaline.py`: O núcleo da Inteligência Artificial. Contém a classe do modelo, inicialização de pesos sinápticos, cálculo da saída linear e o algoritmo de retropropagação do erro.
* `treino.py`: O orquestrador de treinamento. Importa os dados, executa o laço de épocas do Adaline, avalia a acurácia do modelo e utiliza o Matplotlib para desenhar a curva de aprendizagem.
* `interface.py`: O Front-end do sistema. Uma interface que executa o treinamento prévio em background, recebe a imagem escolhida pelo usuário e exibe a letra classificada na tela.

## 🧮 Base Matemática
O aprendizado desta rede Adaline baseia-se na minimização do **Erro Quadrático Médio (MSE)**. O grande diferencial do Adaline é calcular o erro com base na saída contínua (antes de passar pela função de ativação sinal).

A saída linear do neurônio é calculada pelo produto escalar entre os pesos e as entradas, somado ao viés:
$$y = \sum_{i=1}^{n} (w_i \cdot x_i) + b$$

A atualização dos pesos $w$ ocorre através da **Regra Delta**, utilizando uma taxa de aprendizado $\eta$:
$$w_i \leftarrow w_i + \eta \cdot (y_{esperado} - y_{calculado}) \cdot x_i$$

## 🚀 Como Executar

### Pré-requisitos
Certifique-se de ter o Python instalado na sua máquina. Em seguida, instale as dependências executando o comando abaixo no terminal:

```bash
pip install numpy opencv-python pillow matplotlib scikit-learn
