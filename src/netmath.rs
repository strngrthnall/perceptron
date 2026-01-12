use num::pow;

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
pub fn ident(x: f32) -> f32 {
    x
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

pub fn mse(
    out_true: Vec<f32>, 
    out_pred: Vec<f32>, 
    sample_size: usize
) -> f32 {
    let mut sum_squared_errors = 0.0;

    for i in 0..sample_size {
        sum_squared_errors += pow(out_pred[i] - out_true[i], 2);
    }
    sum_squared_errors / sample_size as f32
}
