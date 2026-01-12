/*
 * perceptron.rs
 *
 * Implementação de um perceptron,
 * sem uso de bibliotecas externas de Machine Learning.
 *
 * Este arquivo contém:
 * - Definição da estrutura do neurônio (Perceptron)
 * - Função de ativação identidade
 * - Computação de saída do neurônio
 * - Função de custo MSE (Mean Squared Error)
 * - Cálculo de gradiente por diferenças finitas
 * - Treinamento por gradiente descendente
 *
 * O neurônio aprende a relação linear y = 3x₁ + 2x₂ + 5
 * a partir de exemplos de entrada/saída (2 entradas, 1 saída).
 *
 * Objetivo educacional: mostrar como tudo funciona "por baixo".
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
 * a função linear y = 3x₁ + 2x₂ + 5 (2 entradas, 1 saída).
 *
 * Fluxo de execução:
 *   1. Cria um neurônio com 2 conexões e pesos aleatórios
 *   2. Define dados de treinamento (x₁, x₂, y) da função y = 3x₁ + 2x₂ + 5
 *   3. Exibe o custo inicial (antes do treinamento)
 *   4. Treina o neurônio por 50.000 iterações
 *   5. Exibe o custo final e os parâmetros aprendidos
 *   6. Exibe os dados de teste para verificação
 *
 * Resultado esperado após treinamento:
 *   - weight₁ ≈ 3.0 (coeficiente de x₁)
 *   - weight₂ ≈ 2.0 (coeficiente de x₂)
 *   - bias ≈ 5.0 (termo independente)
 *   - custo ≈ 0 (erro mínimo)
 */
fn main() {
    const SAMPLE_SIZE: usize = 5;

    let mut neuron = init_neuron(ident, 2);

    let x = vec![
        vec![0.0, 0.0], 
        vec![2.0, 15.0], 
        vec![8.0, 3.0], 
        vec![14.0, 18.0],
        vec![20.0, 1.0]
    ]; // Amostras de entrada
    let out_true = vec![5.0, 41.0, 35.0, 83.0, 67.0]; // Saídas esperadas de cada amostra (gabarito)

    // neuron.weights[0] = 2.5;
    // neuron.bias = 6.0;

    let mut cost = compute_cost(&neuron, &x, out_true.clone(), mse, SAMPLE_SIZE);

    println!("***Antes do treinamento***");
    println!("O custo do neurônio : {}", cost);
    println!("O valor do weight 1 : {}", neuron.weights[0]);
    println!("O valor do weight 2 : {}", neuron.weights[1]);
    println!("O valor do bias     : {}", neuron.bias);

    for _i in 0..50000 {
        train(&mut neuron, mse, &x, &out_true, SAMPLE_SIZE);
    }

    cost = compute_cost(&neuron, &x, out_true.clone(), mse, SAMPLE_SIZE);

    println!("***Depois do treinamento***");
    println!("O custo do neurônio : {}", cost);
    println!("O valor do weight 1 : {}", neuron.weights[0]);
    println!("O valor do weight 2 : {}", neuron.weights[1]);
    println!("O valor do bias     : {}", neuron.bias);


    println!("*** Testes ***");
    for i in 0..SAMPLE_SIZE {
        println!("Entradas {} {} - Saída {}", x[i][0], x[i][1], out_true[i]);
    }
    
}