/*
 * neuron.rs
 *
 * Módulo que define a estrutura e operações básicas do neurônio.
 *
 * Este módulo implementa:
 *   - Estrutura Neuron (perceptron)
 *   - Inicialização de neurônios com pesos aleatórios
 *   - Computação da saída do neurônio (forward pass)
 */

use crate::utils::randomize;

/*
 * Estrutura que representa um neurônio (Perceptron).
 *
 * Campos:
 *   weights - vetor de pesos para cada conexão de entrada
 *   n_connections - número de conexões (entradas) do neurônio
 *   bias - termo de viés (deslocamento) adicionado à soma ponderada
 *   act_func - função de ativação aplicada à saída
 *
 * A saída do neurônio é calculada como:
 *   output = act_func(sum(x[i] * weights[i]) + bias)
 */
pub struct Neuron {
    pub weights: Vec<f32>,          // Pesos das conexões
    pub n_connections: u32,         // Número de entradas
    pub bias: f32,                  // Termo de viés
    pub act_func: fn(f32) -> f32    // Função de ativação
}

impl Neuron {
    
    /*
     * Computa o valor de saída do neurônio.
     *
     * Parâmetros:
     *   neuron - neurônio a ser computado
     *   x - vetor de entrada
     *
     * Retorno:
     *   Valor de saída do neurônio
     */
    
    pub fn compute_out(&self, x: &Vec<f32>) -> f32 {
        let mut weighted_sum = 0.0;
        
        for i in 0..self.n_connections {
            weighted_sum += x[i as usize] * self.weights[i as usize];
        }
        weighted_sum += self.bias;
        (self.act_func)(weighted_sum)
    }
    
    /*
     * Cria um neurônio e inicializa seus pesos e bias.
     *
     * Parâmetros:
     *   act_func - a função de ativação do neurônio
     *   n_connections - número de conexões do neurônio
     *
     * Retorno:
     *   O neurônio criado.
     */
    pub fn new(act_func: fn(f32) -> f32, n_connections: u32) -> Self {
        let mut weights: Vec<f32> = Vec::new();
    
        for _i in 0..n_connections {
            weights.push(randomize(-1.0, 1.0));
        }
        
        Self {
            act_func,
            n_connections,
            weights,
            bias: randomize(-1.0, 1.0)
        }
    }
}


