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

pub fn compute_out(neuron: &Neuron, x: &Vec<f32>) -> f32 {
    let mut weighted_sum = 0.0;
    
    for i in 0..neuron.n_connections {
        weighted_sum += x[i as usize] * neuron.weights[i as usize];
    }
    weighted_sum += neuron.bias;
    (neuron.act_func)(weighted_sum)
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

pub fn init_neuron(act_func: fn(f32) -> f32, n_connections: u32) -> Neuron {
    let mut weights: Vec<f32> = Vec::new();

    for _i in 0..n_connections {
        weights.push(randomize(-1.0, 1.0));
    }
    
    Neuron {
        act_func,
        n_connections,
        weights,
        bias: randomize(-1.0, 1.0)
    }
}