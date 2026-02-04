/*
 * main.rs
 *
 * Ponto de entrada da aplicação Perceptron.
 * 
 * Este projeto implementa um perceptron do zero,
 * sem uso de bibliotecas externas de Machine Learning.
 *
 * O projeto está organizado nos seguintes módulos:
 *   - neuron: estrutura do neurônio e funções de inicialização/computação
 *   - neuralnet: funções de treinamento e cálculo de custo
 *   - netmath: funções matemáticas (ativação, MSE)
 *   - utils: utilitários (geração de números aleatórios)
 *
 * O neurônio aprende relações lineares a partir de exemplos
 * de entrada/saída usando gradiente descendente.
 *
 * Objetivo educacional: mostrar como tudo funciona "por baixo do capô".
 */

mod neuron;
mod neuralnet;
mod netmath;
mod utils;

use crate::neuron::*;
use crate::neuralnet::*;
use crate::netmath::*;

/*
 * Função principal - ponto de entrada do programa.
 *
 * Demonstra o treinamento de um perceptron para aprender
 * uma relação linear entre entradas e saídas (2 entradas, 1 saída).
 *
 * Fluxo de execução:
 *   1. Cria um neurônio com 2 conexões e pesos aleatórios
 *   2. Define dados de treinamento (amostras de entrada x e saídas esperadas)
 *   3. Exibe o custo inicial (antes do treinamento)
 *   4. Treina o neurônio por 50.000 iterações usando gradiente descendente
 *   5. Exibe o custo final e os parâmetros aprendidos (pesos e bias)
 *   6. Exibe os resultados de teste para verificação
 *
 * Resultado esperado após treinamento:
 *   - Pesos ajustados para aproximar a relação entre entradas e saídas
 *   - Custo próximo de zero (erro mínimo)
 */
fn main() {
    const SAMPLE_SIZE: usize = 6;
    const CONNECTIONS: u32 = 2;

    let mut neuron = Neuron::new(sigmoid, CONNECTIONS);

    // Dados de treinamento: amostras de entrada (x₁, x₂)
    let x = vec![
        vec![6.0, 1.0], vec![5.0, 0.0], 
        vec![4.0, 1.0], vec![1.0, 4.0],
        vec![1.0, 2.0], vec![2.0, 3.0]
    ];

    // Saídas esperadas (gabarito) correspondentes a cada amostra
    let out_true = vec![
        1.0, 1.0, 
        1.0, 0.0, 
        0.0, 0.0
    ];

    // neuron.weights[0] = 2.5;
    // neuron.bias = 6.0;

    let mut cost = compute_cost(&neuron, &x, &out_true, mse, SAMPLE_SIZE);
    

    println!("***Antes do treinamento***");
    println!("O custo do neurônio : {}", cost);
    for i in 0..CONNECTIONS as usize {
        println!("O valor do weight {} : {}", i+1, neuron.weights[i]);
    }
    println!("O valor do bias     : {}", neuron.bias);

    for _i in 0..50000 {
        train(&mut neuron, mse, &x, &out_true, SAMPLE_SIZE);
    }

    cost = compute_cost(&neuron, &x, &out_true, mse, SAMPLE_SIZE);

    println!("***Depois do treinamento***");
    println!("O custo do neurônio : {}", cost);
    for i in 0..CONNECTIONS as usize {
        println!("O valor do weight {} : {}", i+1, neuron.weights[i]);
    }
    println!("O valor do bias     : {}", neuron.bias);


    println!("*** Testes ***");
    for i in 0..SAMPLE_SIZE {
        println!("Entrada {} {} - Saída {}", x[i][0], x[i][1], neuron.compute_out(&x[i]));
    }
    
}