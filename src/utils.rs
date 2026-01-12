use rand::Rng;

/*
 * Escolhe um valor aleatório entre dois valores.
 *
 * Parâmetros:
 *   min - valor mínimo a ser escolhido
 *   max - valor máximo a ser escolhido
 *
 * Retorno:
 *   Um valor aleatório entre min e max
 */

pub fn randomize(min: f32, max: f32) -> f32 {
    rand::thread_rng().gen_range(min..max)
}