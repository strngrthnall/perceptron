use rand::Rng;
use num::pow;

/*
 * perceptron.rs
 *
 * Implementação de um perceptron,
 * sem uso de bibliotecas externas.
 *
 * Este arquivo contém:
 * - definição da estrutura do perceptron
 * - função de custo e computação do neurônio
 *
 * Objetivo educacional: mostrar como tudo funciona "por baixo".
 */

struct Neuron {
    weights: Vec<f32>,          // float *weight;
    n_connections: u32,         // uint32_t nconnections;
    bias: f32,                  // float bias;
    act_func: fn(f32) -> f32    // float (*actfunc)();
}


/*
 * Escolhe um valor randômico entre dois valores
 *
 * Parâmetros:
 *   min - valor minimo a ser escolhido
 *   max - valor máximo a ser escolhido
 *
 * Retorno
 *   Um valor randômico entre min e max 
 */

fn randomize(min: f32, max: f32) -> f32 {
    rand::thread_rng().gen_range(min..max)
}

/*
 * Calcula a função de custo Identidade.
 *
 * Parâmetros:
 *   x - Um valor escalar
 *
 * Retorno:
 *   valor do custo da rede neural
 */

fn ident(x: f32) -> f32 {
    x
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

fn comput_out(neuron: &Neuron, x: &Vec<f32>) -> f32 {
    let mut k = 0.0;
    
    for i in 0..neuron.n_connections {
        k += x[i as usize] * neuron.weights[i as usize];
    }
    k += neuron.bias;
    (neuron.act_func)(k)
}

/*
 * Cria um neurônio, inicializa seu pesos e bia
 *
 * Parâmetros:
 *   act_func - a função de ativação do neurônio
 *   nconnections - número de conexões do neurônio
 *
 * Retorno
 *   O neurônio criado.
 */

fn init_neuron(act_func: fn(f32) -> f32, n_connections: u32) -> Neuron {
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

fn mse(out_true: Vec<f32>, out_pred: Vec<f32>, sample_size: usize) -> f32 {
    let mut s = 0.0;

    for i in 0..sample_size {
        s += pow(out_pred[i] - out_true[i], 2);
    }
    s / sample_size as f32
}

fn main() {

    let mut neuron = init_neuron(ident, 1);

    let x = vec![
        vec![0.0], 
        vec![2.0], 
        vec![4.0], 
        vec![6.0]
    ];
    let out_true = vec![6.0, 11.0, 16.0, 21.0];
    let mut out_pred: Vec<f32> = vec![];

    neuron.weights[0] = 2.5;
    neuron.bias = 6.0;

    for i in 0..4 {
        out_pred.push(comput_out(&neuron, &x[i]));
    }

    println!("O valor do wheight é: {}", neuron.weights[0]);
    println!("O valor do bias é: {}", neuron.bias);

    println!("O custo do neurônio é: {}", mse(out_true, out_pred, 4))
    
}