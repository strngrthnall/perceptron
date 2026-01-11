<p align="center">
  <img src="https://img.shields.io/badge/Rust-000000?style=for-the-badge&logo=rust&logoColor=white" alt="Rust">
  <img src="https://img.shields.io/badge/Machine%20Learning-FF6F00?style=for-the-badge&logo=tensorflow&logoColor=white" alt="Machine Learning">
  <img src="https://img.shields.io/badge/Status-Em%20Desenvolvimento-yellow?style=for-the-badge" alt="Status">
</p>

<h1 align="center">🧠 Perceptron em Rust</h1>

<p align="center">
  <strong>Implementação de um Perceptron do zero, sem frameworks de Machine Learning</strong>
</p>

<p align="center">
  <a href="#-sobre">Sobre</a> •
  <a href="#-motivação">Motivação</a> •
  <a href="#-funcionalidades">Funcionalidades</a> •
  <a href="#-estrutura">Estrutura</a> •
  <a href="#-como-executar">Como Executar</a> •
  <a href="#-referências">Referências</a> •
  <a href="#-licença">Licença</a>
</p>

---

## 📖 Sobre

Este projeto é uma implementação de um **Perceptron** em **Rust**, baseado nas aulas do canal **[Do Zero](https://www.youtube.com/@dozero)** no YouTube, onde a implementação original é feita em **C**.

O objetivo principal é **aprender os fundamentos de redes neurais** construindo tudo do zero, sem depender de bibliotecas de Machine Learning como TensorFlow ou PyTorch. Aqui, optamos por reescrever o projeto em **Rust** para explorar as vantagens da linguagem em termos de segurança de memória e performance.

### Como funciona

O perceptron implementado é capaz de aprender funções lineares através do algoritmo de **gradiente descendente**:

1. **Inicialização**: Pesos e bias são inicializados com valores aleatórios
2. **Forward Pass**: Calcula a saída do neurônio: `y = f(Σ(xᵢ × wᵢ) + bias)`
3. **Cálculo do Custo**: Mede o erro usando MSE (Mean Squared Error)
4. **Cálculo do Gradiente**: Usa diferenças finitas para aproximar a derivada
5. **Atualização**: Ajusta pesos e bias na direção que reduz o erro

> ⚠️ **Nota:** Este é um projeto de **estudo** e não deve ser utilizado em produção. O foco está no aprendizado dos conceitos fundamentais de redes neurais artificiais.

---

## 🎯 Motivação

- 📚 **Aprendizado**: Compreender os conceitos fundamentais de redes neurais na prática
- 🦀 **Rust**: Praticar a linguagem Rust em um contexto de Machine Learning
- 🔧 **Do Zero**: Implementar sem abstrações para entender "por baixo do capô"
- 🎥 **Inspiração**: Acompanhar e adaptar o conteúdo do canal Do Zero para Rust

---

## ✨ Funcionalidades

- [x] Estrutura básica do Neurônio (Perceptron)
- [x] Inicialização de pesos e bias aleatórios
- [x] Função de ativação (Identidade)
- [x] Computação de saída do neurônio
- [x] Função de custo MSE (Mean Squared Error)
- [x] Cálculo de gradiente por diferenças finitas
- [x] Algoritmo de treinamento (Gradiente Descendente)
- [ ] Múltiplas funções de ativação (Sigmoid, ReLU, Tanh)
- [ ] Múltiplas camadas (MLP - Multi-Layer Perceptron)

---

## 📁 Estrutura

```
perceptron/
├── Cargo.toml          # Configuração do projeto e dependências
├── README.md           # Documentação do projeto
└── src/
    └── main.rs         # Implementação do perceptron
```

### Componentes Principais

| Componente | Descrição |
|------------|-----------|
| `Neuron` | Estrutura que representa um neurônio com pesos, bias e função de ativação |
| `init_neuron()` | Inicializa um neurônio com pesos e bias aleatórios |
| `comput_out()` | Calcula a saída do neurônio dado um vetor de entrada |
| `mse()` | Calcula o erro quadrático médio (Mean Squared Error) |
| `comput_cost()` | Calcula o custo total do neurônio para um conjunto de amostras |
| `comput_gradient()` | Calcula o gradiente de um parâmetro usando diferenças finitas |
| `train()` | Treina o neurônio usando gradiente descendente |
| `randomize()` | Gera valores aleatórios em um intervalo |

---

## 🚀 Como Executar

### Pré-requisitos

- [Rust](https://www.rust-lang.org/tools/install) instalado (versão 1.70+)
- Cargo (gerenciador de pacotes do Rust)

### Instalação

```bash
# Clone o repositório
git clone https://github.com/strngrthnall/perceptron.git

# Entre no diretório
cd perceptron

# Compile o projeto
cargo build --release

# Execute
cargo run
```

### Saída Esperada

O programa treina um neurônio para aprender a função linear `y = 2.5x + 6`:

```
***Antes do treinamento***
O custo do neurônio : 248.5    (valor aleatório)
O valor do wheight  : 0.42     (peso aleatório)
O valor do bias     : -0.78    (bias aleatório)

***Depois do treinamento***
O custo do neurônio : ~0       (erro mínimo)
O valor do wheight  : ~2.5     (coeficiente angular aprendido)
O valor do bias     : ~6.0     (termo independente aprendido)
```

> 💡 Os valores iniciais são aleatórios, mas após 50.000 iterações de treinamento,
> o neurônio converge para os parâmetros corretos da função `y = 2.5x + 6`.

---

## 📚 Referências

- 🎥 **Canal Do Zero** - [YouTube](https://www.youtube.com/@dozero)
  - Série de vídeos sobre implementação de redes neurais em C
- 📖 **Documentação Rust** - [rust-lang.org](https://doc.rust-lang.org/book/)
- 🧠 **Perceptron** - [Wikipedia](https://en.wikipedia.org/wiki/Perceptron)

---

## 🛠️ Tecnologias

| Tecnologia | Versão | Uso |
|------------|--------|-----|
| Rust | 2024 Edition | Linguagem principal |
| rand | 0.8 | Geração de números aleatórios |
| num | 0.4.3 | Operações matemáticas |

---

## 📝 Licença

Este projeto está sob a licença MIT. Veja o arquivo [LICENSE](LICENSE) para mais detalhes.

---

## 🤝 Contribuição

Contribuições são bem-vindas! Este é um projeto de estudo, então sinta-se à vontade para:

1. Fazer um Fork do projeto
2. Criar uma branch para sua feature (`git checkout -b feature/AmazingFeature`)
3. Commit suas mudanças (`git commit -m 'Add some AmazingFeature'`)
4. Push para a branch (`git push origin feature/AmazingFeature`)
5. Abrir um Pull Request

---

<p align="center">
  Feito com ❤️ para fins educacionais
</p>

<p align="center">
  <sub>Inspirado nas aulas do canal <a href="https://www.youtube.com/@dozero">Do Zero</a></sub>
</p>
