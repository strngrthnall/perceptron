use rand::Rng;
use num::pow;

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
 * O neurônio aprende a relação linear y = 2.5x + 6
 * a partir de exemplos de entrada/saída.
 *
 * Objetivo educacional: mostrar como tudo funciona "por baixo".
 */

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
struct Neuron {
    weights: Vec<f32>,          // Pesos das conexões
    n_connections: u32,         // Número de entradas
    bias: f32,                  // Termo de viés
    act_func: fn(f32) -> f32    // Função de ativação
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
 * Função de ativação Identidade.
 *
 * Retorna o valor de entrada sem modificação.
 * Útil para problemas de regressão linear onde
 * a saída não precisa ser limitada a um intervalo.
 *
 * Parâmetros:
 *   x - valor de entrada (soma ponderada + bias)
 *
 * Retorno:
 *   O próprio valor x (f(x) = x)
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

/*
 * Calcula o Erro Quadrático Médio (Mean Squared Error).
 *
 * Parâmetros:
 *   out_true - vetor com os valores esperados (gabarito)
 *   out_pred - vetor com os valores preditos pelo neurônio
 *   sample_size - número de amostras
 *
 * Retorno:
 *   O erro quadrático médio entre os valores esperados e preditos
 */

fn mse(
    out_true: Vec<f32>, 
    out_pred: Vec<f32>, 
    sample_size: usize
) -> f32 {
    let mut s = 0.0;

    for i in 0..sample_size {
        s += pow(out_pred[i] - out_true[i], 2);
    }
    s / sample_size as f32
}

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

fn comput_cost(
    neuron: &Neuron, 
    x: &Vec<Vec<f32>>, 
    y: Vec<f32>, 
    cost: fn(Vec<f32>, Vec<f32>, usize) -> f32, 
    sample_size: usize
) -> f32 {
    let mut out_pred: Vec<f32> = Vec::new();

    for i in 0..sample_size {
        out_pred.push(comput_out(&neuron, &x[i]));
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
fn comput_gradient(
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
    let variation_cost = comput_cost(neuron, x, y.clone(), cost, sample_size);
    
    // Restaura o parâmetro
    match param {
        ParamType::Weight(i) => neuron.weights[i] -= eps,
        ParamType::Bias => neuron.bias -= eps,
    }
    let normal_cost = comput_cost(neuron, x, y.clone(), cost, sample_size);

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

fn train(
    neuron: &mut Neuron, 
    cost: fn(Vec<f32>, Vec<f32>, usize) -> f32, 
    x: &Vec<Vec<f32>>, 
    y: &Vec<f32>, 
    sample_size: usize
) {

    let mut gradient;

    for i in 0..neuron.n_connections as usize {
        let param = ParamType::Weight(i);
        gradient = comput_gradient(neuron, cost, x, y, param, sample_size);
        neuron.weights[i] -= 0.001 * gradient;
    }

    let param = ParamType::Bias;
    gradient = comput_gradient(neuron, cost, x, y, param, sample_size);
    neuron.bias -= 0.001 * gradient;


}

/*
 * Função principal - ponto de entrada do programa.
 *
 * Demonstra o treinamento de um perceptron para aprender
 * a função linear y = 2.5x + 6.
 *
 * Fluxo de execução:
 *   1. Cria um neurônio com 1 conexão e pesos aleatórios
 *   2. Define dados de treinamento (x, y) da função y = 2.5x + 6
 *   3. Exibe o custo inicial (antes do treinamento)
 *   4. Treina o neurônio por 50.000 iterações
 *   5. Exibe o custo final e os parâmetros aprendidos
 *
 * Resultado esperado após treinamento:
 *   - weight ≈ 2.5 (coeficiente angular)
 *   - bias ≈ 6.0 (termo independente)
 *   - custo ≈ 0 (erro mínimo)
 */
fn main() {

    let mut neuron = init_neuron(ident, 2);

    let x = vec![
        vec![0.0], 
        vec![2.0], 
        vec![4.0], 
        vec![6.0]
    ]; // Amostras de entrada
    let out_true = vec![6.0, 11.0, 16.0, 21.0]; // Saídas esperada de cada amostra (gabarito)

    // neuron.weights[0] = 2.5;
    // neuron.bias = 6.0;

    let mut cost = comput_cost(&neuron, &x, out_true.clone(), mse, 4);

    println!("***Antes do treinamento***");
    println!("O custo do neurônio : {}", cost);

    println!("O valor do wheight  : {}", neuron.weights[0]);
    println!("O valor do bias     : {}", neuron.bias);

    for _i in 0..50000 {
        train(&mut neuron, mse, &x, &out_true, 4);
    }

    cost = comput_cost(&neuron, &x, out_true.clone(), mse, 4);

    println!("***Depois do treinamento***");
    println!("O custo do neurônio : {}", cost);
    println!("O valor do wheight  : {}", neuron.weights[0]);
    println!("O valor do bias     : {}", neuron.bias);


    println!("*** Testes ***");
    for i in 0..4 {
        println!("Entrada {} - Saída {}", x[i][0], out_true[i]);
    }
    
}