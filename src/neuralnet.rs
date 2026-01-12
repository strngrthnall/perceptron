use crate::neuron::*;

/*
 * Calcula o custo total do neurônio para um conjunto de amostras.
 *
 * Parâmetros:
 *   neuron - referência ao neurônio a ser avaliado
 *   x - vetor de vetores contendo as entradas de cada amostra
 *   y - vetor com os valores esperados (gabarito)
 *   cost - função de custo a ser utilizada (ex: mse)
 *   sample_size - número de amostras
 *
 * Retorno:
 *   O custo calculado pela função de custo fornecida
 */

pub fn compute_cost(
    neuron: &Neuron, 
    x: &Vec<Vec<f32>>, 
    y: Vec<f32>, 
    cost: fn(Vec<f32>, Vec<f32>, usize) -> f32, 
    sample_size: usize
) -> f32 {
    let mut out_pred: Vec<f32> = Vec::new();

    for i in 0..sample_size {
        out_pred.push(compute_out(&neuron, &x[i]));
    }
    cost(y, out_pred, sample_size)
}

/*
 * Enum que representa o tipo de parâmetro a ser ajustado.
 *
 * Variantes:
 *   Weight(usize) - representa um peso específico pelo seu índice
 *   Bias - representa o bias do neurônio
 */

enum ParamType {
    Weight(usize),
    Bias,
}
/*
 * Calcula o gradiente de um parâmetro usando diferenças finitas.
 *
 * Utiliza a aproximação numérica da derivada:
 *   lim (delta -> 0) [cost(param + delta) - cost(param)] / delta
 *
 * Parâmetros:
 *   neuron - referência mutável ao neurônio
 *   cost - função de custo a ser utilizada (ex: mse)
 *   x - vetor de vetores contendo as entradas de cada amostra
 *   y - vetor com os valores esperados (gabarito)
 *   param - tipo do parâmetro a ser derivado (Weight ou Bias)
 *   sample_size - número de amostras
 *
 * Retorno:
 *   O gradiente (derivada parcial) do parâmetro em relação ao custo
 */
fn compute_gradient(
    neuron: &mut Neuron, 
    cost: fn(Vec<f32>, Vec<f32>, usize) -> f32,
    x: &Vec<Vec<f32>>,
    y: &Vec<f32>,
    param: ParamType,
    sample_size: usize
) -> f32 {
    let eps = 0.0001;
    
    // Modifica o parâmetro diretamente no neurônio
    match param {
        ParamType::Weight(i) => neuron.weights[i] += eps,
        ParamType::Bias => neuron.bias += eps,
    }
    let variation_cost = compute_cost(neuron, x, y.clone(), cost, sample_size);
    
    // Restaura o parâmetro
    match param {
        ParamType::Weight(i) => neuron.weights[i] -= eps,
        ParamType::Bias => neuron.bias -= eps,
    }
    let normal_cost = compute_cost(neuron, x, y.clone(), cost, sample_size);

    (variation_cost - normal_cost) / eps
}

/*
 * Treina o neurônio ajustando seus pesos e bias.
 *
 * Utiliza o algoritmo de gradiente descendente para minimizar
 * a função de custo, atualizando os parâmetros iterativamente.
 *
 * Parâmetros:
 *   neuron - referência ao neurônio a ser treinado
 *   cost - função de custo a ser minimizada (ex: mse)
 *   x - vetor de vetores contendo as entradas de cada amostra
 *   y - vetor com os valores esperados (gabarito)
 *   sample_size - número de amostras
 *
 * Retorno:
 *   Nenhum (modifica o neurônio in-place)
 */

pub fn train(
    neuron: &mut Neuron, 
    cost: fn(Vec<f32>, Vec<f32>, usize) -> f32, 
    x: &Vec<Vec<f32>>, 
    y: &Vec<f32>, 
    sample_size: usize
) {
    let mut gradient;

    for i in 0..neuron.n_connections as usize {
        let param = ParamType::Weight(i);
        gradient = compute_gradient(neuron, cost, x, y, param, sample_size);
        neuron.weights[i] -= 0.001 * gradient;
    }

    let param = ParamType::Bias;
    gradient = compute_gradient(neuron, cost, x, y, param, sample_size);
    neuron.bias -= 0.001 * gradient;

}